import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def clamp_reputation_confidence(trust_matrix: torch.Tensor) -> torch.Tensor:
    trust_matrix[:, 0] = trust_matrix[:, 0].clamp(-1.0, 1.0)
    trust_matrix[:, 1] = trust_matrix[:, 1].clamp(0.0, 1.0)
    return trust_matrix


@dataclass
class BeliefState:
    trust_matrix: torch.Tensor  # [N, 2] => [reputation, confidence]
    incoming_proposals: torch.Tensor  # [N] binary

    def vectorize(self) -> torch.Tensor:
        return torch.cat(
            [self.trust_matrix.reshape(-1), self.incoming_proposals.to(torch.float32)], dim=0
        )


@dataclass(frozen=True)
class PayoffConfig:
    mutual_befriend: Tuple[float, float] = (100.0, 100.0)  # (1, 1)
    betray_befriend: Tuple[float, float] = (103.0, -106.0)  # (0, 1)
    befriend_betray: Tuple[float, float] = (-106.0, 103.0)  # (1, 0)
    mutual_betray: Tuple[float, float] = (-2.0, -2.0)  # (0, 0)
    isolation: float = -1.0


class MiniSocietyEnv:
    """Iterated Prisoner's Dilemma with partner selection, reputation, and memory decay."""

    def __init__(
        self,
        n_agents: int = 5,
        episode_length: int = 64,
        confidence_decay: float = 0.005,
        gossip_noise: float = 0.04,
        gossip_strength: float = 0.18,
        initial_confidence: float = 0.15,
        reputation_delta_befriend: float = 0.15,
        reputation_delta_betray: float = -0.35,
        confidence_gain_interaction: float = 1.0,
        payoff: Optional[PayoffConfig] = None,
        device: str = "cpu",
    ) -> None:
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2.")
        if episode_length < 0:
            raise ValueError("episode_length must be non-negative (0 means unbounded).")
        if gossip_noise < 0:
            raise ValueError("gossip_noise must be non-negative.")
        if not (0.0 <= gossip_strength <= 1.0):
            raise ValueError("gossip_strength must be in [0, 1].")
        if confidence_decay < 0:
            raise ValueError("confidence_decay must be non-negative.")
        if not (0.0 <= initial_confidence <= 1.0):
            raise ValueError("initial_confidence must be in [0, 1].")
        if not (0.0 <= confidence_gain_interaction <= 1.0):
            raise ValueError("confidence_gain_interaction must be in [0, 1].")
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.confidence_decay = confidence_decay
        self.gossip_noise = gossip_noise
        self.gossip_strength = gossip_strength
        self.initial_confidence = initial_confidence
        self.reputation_delta_befriend = reputation_delta_befriend
        self.reputation_delta_betray = reputation_delta_betray
        self.confidence_gain_interaction = confidence_gain_interaction
        self.payoff = payoff or PayoffConfig()
        self.payoff_matrix = {
            (1, 1): self.payoff.mutual_befriend,
            (0, 1): self.payoff.betray_befriend,
            (1, 0): self.payoff.befriend_betray,
            (0, 0): self.payoff.mutual_betray,
        }
        self.initial_reputation_prior: Optional[torch.Tensor] = None
        self.initial_confidence_prior: Optional[torch.Tensor] = None
        self.device = torch.device(device)
        self.timestep = 0
        self.states: List[BeliefState] = []
        self.reset()

    @property
    def obs_dim(self) -> int:
        # Per target: [my reputation, my confidence, incoming proposal, target's reputation of me, target's confidence in me]
        return self.n_agents * 5

    def _initial_state(self, observer_id: int) -> BeliefState:
        trust = torch.zeros((self.n_agents, 2), device=self.device, dtype=torch.float32)
        if self.initial_reputation_prior is not None:
            trust[:, 0] = self.initial_reputation_prior[observer_id].to(self.device)
        if self.initial_confidence_prior is not None:
            trust[:, 1] = self.initial_confidence_prior[observer_id].to(self.device)
        else:
            trust[:, 1] = self.initial_confidence
        clamp_reputation_confidence(trust)
        incoming = torch.zeros(self.n_agents, device=self.device, dtype=torch.float32)
        return BeliefState(trust_matrix=trust, incoming_proposals=incoming)

    def reset(self) -> List[torch.Tensor]:
        self.timestep = 0
        self.states = [self._initial_state(i) for i in range(self.n_agents)]
        return [self.observe(i) for i in range(self.n_agents)]

    def set_initial_priors(
        self,
        reputation_prior: Optional[torch.Tensor],
        confidence_prior: Optional[torch.Tensor],
    ) -> None:
        if reputation_prior is not None:
            if reputation_prior.shape != (self.n_agents, self.n_agents):
                raise ValueError(
                    f"reputation_prior must have shape {(self.n_agents, self.n_agents)}; got {tuple(reputation_prior.shape)}"
                )
            self.initial_reputation_prior = reputation_prior.detach().to(torch.float32).clone()
        else:
            self.initial_reputation_prior = None

        if confidence_prior is not None:
            if confidence_prior.shape != (self.n_agents, self.n_agents):
                raise ValueError(
                    f"confidence_prior must have shape {(self.n_agents, self.n_agents)}; got {tuple(confidence_prior.shape)}"
                )
            self.initial_confidence_prior = confidence_prior.detach().to(torch.float32).clone().clamp(0.0, 1.0)
        else:
            self.initial_confidence_prior = None

    def get_state(self) -> Dict[str, object]:
        trust_stack = torch.stack([s.trust_matrix.detach().cpu().clone() for s in self.states], dim=0)
        incoming_stack = torch.stack([s.incoming_proposals.detach().cpu().clone() for s in self.states], dim=0)
        return {
            "n_agents": self.n_agents,
            "episode_length": self.episode_length,
            "timestep": self.timestep,
            "trust_stack": trust_stack,
            "incoming_stack": incoming_stack,
            "initial_reputation_prior": None
            if self.initial_reputation_prior is None
            else self.initial_reputation_prior.detach().cpu().clone(),
            "initial_confidence_prior": None
            if self.initial_confidence_prior is None
            else self.initial_confidence_prior.detach().cpu().clone(),
        }

    def load_state(self, state: Dict[str, object]) -> None:
        n_agents = int(state["n_agents"])
        if n_agents != self.n_agents:
            raise ValueError(f"Checkpoint has n_agents={n_agents}, current env has n_agents={self.n_agents}.")
        self.episode_length = int(state["episode_length"])
        self.timestep = int(state["timestep"])
        trust_stack = state["trust_stack"].to(self.device)
        incoming_stack = state["incoming_stack"].to(self.device)
        self.states = []
        for i in range(self.n_agents):
            self.states.append(
                BeliefState(
                    trust_matrix=trust_stack[i].to(torch.float32).clone(),
                    incoming_proposals=incoming_stack[i].to(torch.float32).clone(),
                )
            )
        self.set_initial_priors(
            reputation_prior=state["initial_reputation_prior"],
            confidence_prior=state["initial_confidence_prior"],
        )

    def observe(self, agent_id: int) -> torch.Tensor:
        outgoing_trust = self.states[agent_id].trust_matrix.reshape(-1)
        incoming = self.states[agent_id].incoming_proposals.to(torch.float32)
        reciprocal_trust = torch.stack(
            [self.states[target].trust_matrix[agent_id] for target in range(self.n_agents)],
            dim=0,
        ).reshape(-1)
        return torch.cat([outgoing_trust, incoming, reciprocal_trust], dim=0)

    def trust_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reputation = torch.stack([s.trust_matrix[:, 0].detach().clone() for s in self.states], dim=0)
        confidence = torch.stack([s.trust_matrix[:, 1].detach().clone() for s in self.states], dim=0)
        return reputation, confidence

    def proposal_mask(self, agent_id: int) -> torch.Tensor:
        mask = torch.ones(self.n_agents + 1, device=self.device, dtype=torch.bool)
        mask[agent_id] = False
        return mask

    def response_mask(self, agent_id: int) -> torch.Tensor:
        incoming = self.states[agent_id].incoming_proposals.bool()
        mask = torch.zeros(self.n_agents + 1, device=self.device, dtype=torch.bool)
        mask[: self.n_agents] = incoming
        mask[self.n_agents] = True  # always allowed to decline all
        return mask

    def _apply_confidence_decay(self) -> None:
        for state in self.states:
            state.trust_matrix[:, 1] = (state.trust_matrix[:, 1] * (1.0 - self.confidence_decay)).clamp(0.0, 1.0)

    def _build_incoming_proposals(self, proposals: Sequence[int]) -> None:
        for receiver in range(self.n_agents):
            incoming = torch.zeros(self.n_agents, device=self.device, dtype=torch.float32)
            for proposer, target in enumerate(proposals):
                if proposer == receiver:
                    continue
                if target == receiver:
                    incoming[proposer] = 1.0
            self.states[receiver].incoming_proposals = incoming

    def _set_play_context_incoming(self, pairs: Sequence[Tuple[int, int]]) -> None:
        """Expose partner identity to the play head via incoming_proposals one-hot context."""
        for agent_a, agent_b in pairs:
            incoming_a = torch.zeros(self.n_agents, device=self.device, dtype=torch.float32)
            incoming_b = torch.zeros(self.n_agents, device=self.device, dtype=torch.float32)
            incoming_a[agent_b] = 1.0
            incoming_b[agent_a] = 1.0
            self.states[agent_a].incoming_proposals = incoming_a
            self.states[agent_b].incoming_proposals = incoming_b

    def _form_pairs(self, proposals: Sequence[int], responses: Sequence[int]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        matched = set()
        for proposer, target in enumerate(proposals):
            if target >= self.n_agents:
                continue
            if proposer in matched or target in matched:
                continue
            if responses[target] == proposer:
                matched.add(proposer)
                matched.add(target)
                pairs.append((proposer, target))
        return pairs

    def _apply_pairwise_reputation_update(
        self, agent_a: int, agent_b: int, action_a: int, action_b: int
    ) -> None:
        # action: 0=betray, 1=befriend
        delta_a = self.reputation_delta_befriend if action_a == 1 else self.reputation_delta_betray
        delta_b = self.reputation_delta_befriend if action_b == 1 else self.reputation_delta_betray

        # a updates belief about b and vice-versa
        self.states[agent_a].trust_matrix[agent_b, 0] += delta_b
        self.states[agent_b].trust_matrix[agent_a, 0] += delta_a
        self.states[agent_a].trust_matrix[agent_b, 1] = max(
            float(self.states[agent_a].trust_matrix[agent_b, 1].item()),
            self.confidence_gain_interaction,
        )
        self.states[agent_b].trust_matrix[agent_a, 1] = max(
            float(self.states[agent_b].trust_matrix[agent_a, 1].item()),
            self.confidence_gain_interaction,
        )

        clamp_reputation_confidence(self.states[agent_a].trust_matrix)
        clamp_reputation_confidence(self.states[agent_b].trust_matrix)

    @staticmethod
    def _blend_trust_evidence(
        current_rep: float,
        current_conf: float,
        evidence_rep: float,
        evidence_conf: float,
    ) -> Tuple[float, float]:
        bounded_current_conf = max(0.0, min(1.0, current_conf))
        bounded_evidence_conf = max(0.0, min(1.0, evidence_conf))
        if bounded_evidence_conf <= 0.0:
            return current_rep, bounded_current_conf
        weight = bounded_evidence_conf / (bounded_current_conf + bounded_evidence_conf + 1e-8)
        blended_rep = (1.0 - weight) * current_rep + weight * evidence_rep
        blended_conf = bounded_current_conf + bounded_evidence_conf * (1.0 - bounded_current_conf)
        return float(max(-1.0, min(1.0, blended_rep))), float(max(0.0, min(1.0, blended_conf)))

    def _gossip_blend(self, agent_a: int, agent_b: int) -> None:
        if self.gossip_strength <= 0.0:
            return

        trust_a = self.states[agent_a].trust_matrix
        trust_b = self.states[agent_b].trust_matrix
        prev_a = trust_a.clone()
        prev_b = trust_b.clone()
        max_gossip_conf = self.confidence_gain_interaction * self.gossip_strength

        for target in range(self.n_agents):
            if target in (agent_a, agent_b):
                continue

            shared_conf_to_a = min(float(prev_b[target, 1].item()) * self.gossip_strength, max_gossip_conf)
            shared_conf_to_b = min(float(prev_a[target, 1].item()) * self.gossip_strength, max_gossip_conf)

            if shared_conf_to_a > 0.0:
                heard_rep_a = float(prev_b[target, 0].item()) + random.uniform(-self.gossip_noise, self.gossip_noise)
                current_conf_a = float(prev_a[target, 1].item())
                updated_rep_a, updated_conf_a = self._blend_trust_evidence(
                    current_rep=float(prev_a[target, 0].item()),
                    current_conf=current_conf_a,
                    evidence_rep=heard_rep_a,
                    evidence_conf=shared_conf_to_a,
                )
                trust_a[target, 0] = updated_rep_a
                trust_a[target, 1] = updated_conf_a

            if shared_conf_to_b > 0.0:
                heard_rep_b = float(prev_a[target, 0].item()) + random.uniform(-self.gossip_noise, self.gossip_noise)
                current_conf_b = float(prev_b[target, 1].item())
                updated_rep_b, updated_conf_b = self._blend_trust_evidence(
                    current_rep=float(prev_b[target, 0].item()),
                    current_conf=current_conf_b,
                    evidence_rep=heard_rep_b,
                    evidence_conf=shared_conf_to_b,
                )
                trust_b[target, 0] = updated_rep_b
                trust_b[target, 1] = updated_conf_b

        clamp_reputation_confidence(trust_a)
        clamp_reputation_confidence(trust_b)

    def _resolve_step(
        self,
        pairs: Sequence[Tuple[int, int]],
        proposals: Sequence[int],
        responses: Sequence[int],
        play_actions: Dict[int, int],
    ) -> Tuple[List[torch.Tensor], List[float], bool, Dict[str, object]]:
        # Age prior beliefs before incorporating new evidence so direct experience
        # can land at full confidence for the current interaction step.
        self._apply_confidence_decay()
        rewards = [float(self.payoff.isolation) for _ in range(self.n_agents)]  # isolation default
        interaction_outcomes: List[Dict[str, object]] = []
        befriend_count = 0
        betray_count = 0

        for agent_a, agent_b in pairs:
            action_a = int(play_actions[agent_a])
            action_b = int(play_actions[agent_b])

            reward_a, reward_b = self.payoff_matrix[(action_a, action_b)]
            if action_a == 1 and action_b == 1:
                outcome = "mutual_befriend"
            elif action_a == 0 and action_b == 0:
                outcome = "mutual_betray"
            elif action_a == 0 and action_b == 1:
                outcome = "betray_befriend"
            else:
                outcome = "befriend_betray"

            rewards[agent_a] = reward_a
            rewards[agent_b] = reward_b
            befriend_count += int(action_a == 1) + int(action_b == 1)
            betray_count += int(action_a == 0) + int(action_b == 0)
            interaction_outcomes.append(
                {
                    "agents": (agent_a, agent_b),
                    "actions": (action_a, action_b),
                    "rewards": (float(reward_a), float(reward_b)),
                    "outcome": outcome,
                }
            )
            self._apply_pairwise_reputation_update(agent_a, agent_b, action_a, action_b)
            # Agents only exchange third-party trust information after mutual cooperation.
            if action_a == 1 and action_b == 1:
                self._gossip_blend(agent_a, agent_b)

        reputation, confidence = self.trust_snapshot()

        for state in self.states:
            state.incoming_proposals.zero_()

        self.timestep += 1
        done = self.episode_length > 0 and self.timestep >= self.episode_length
        info: Dict[str, object] = {
            "pairs": list(pairs),
            "proposals": list(proposals),
            "responses": list(responses),
            "play_actions": dict(play_actions),
            "interaction_outcomes": interaction_outcomes,
            "reputation": reputation,
            "confidence": confidence,
            "societal_reward": float(sum(rewards)),
            "befriend_count": befriend_count,
            "betray_count": betray_count,
        }
        return [self.observe(i) for i in range(self.n_agents)], rewards, done, info

    def step(
        self, agents: Sequence["DecentralizedPPOAgent"]
    ) -> Tuple[List[torch.Tensor], List[float], bool, Dict[str, object], List["RolloutStep"]]:
        if len(agents) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agents, got {len(agents)}")

        propose_obs = [self.observe(i).to(self.device) for i in range(self.n_agents)]
        proposals: List[int] = []
        logp_prop: List[float] = []
        proposal_masks: List[torch.Tensor] = []
        value_est: List[float] = []

        for i in range(self.n_agents):
            proposal_mask = self.proposal_mask(i)
            action, logp, _, value = agents[i].act_propose(propose_obs[i], proposal_mask)
            proposals.append(action)
            logp_prop.append(logp)
            proposal_masks.append(proposal_mask.clone())
            value_est.append(value)

        self._build_incoming_proposals(proposals)

        respond_obs = [self.observe(i).to(self.device) for i in range(self.n_agents)]
        responses: List[int] = []
        respond_masks: List[torch.Tensor] = []
        logp_resp: List[float] = []

        for i in range(self.n_agents):
            mask = self.response_mask(i)
            action, logp, _ = agents[i].act_respond(respond_obs[i], mask)
            responses.append(action)
            logp_resp.append(logp)
            respond_masks.append(mask.clone())

        pairs = self._form_pairs(proposals, responses)
        self._set_play_context_incoming(pairs)
        paired_agents = {agent for pair in pairs for agent in pair}
        play_actions: Dict[int, int] = {}
        play_obs: List[torch.Tensor] = [
            torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device) for _ in range(self.n_agents)
        ]
        play_action: List[int] = [-1 for _ in range(self.n_agents)]
        play_logp: List[float] = [0.0 for _ in range(self.n_agents)]
        played: List[bool] = [False for _ in range(self.n_agents)]

        for agent_id in paired_agents:
            obs = self.observe(agent_id).to(self.device)
            action, logp, _ = agents[agent_id].act_play(obs)
            play_actions[agent_id] = action
            play_obs[agent_id] = obs
            play_action[agent_id] = action
            play_logp[agent_id] = logp
            played[agent_id] = True

        next_obs, rewards, done, info = self._resolve_step(pairs, proposals, responses, play_actions)

        rollout_steps: List[RolloutStep] = []
        for i in range(self.n_agents):
            rollout_steps.append(
                RolloutStep(
                    obs_propose=propose_obs[i].detach().clone(),
                    action_propose=proposals[i],
                    logprob_propose=logp_prop[i],
                    proposal_mask=proposal_masks[i].detach().clone(),
                    obs_respond=respond_obs[i].detach().clone(),
                    action_respond=responses[i],
                    logprob_respond=logp_resp[i],
                    respond_mask=respond_masks[i].detach().clone(),
                    obs_play=play_obs[i].detach().clone(),
                    action_play=play_action[i],
                    logprob_play=play_logp[i],
                    played=played[i],
                    value=value_est[i],
                    reward=float(rewards[i]),
                    done=done,
                )
            )

        return next_obs, rewards, done, info, rollout_steps


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.target_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.context_backbone = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.propose_target_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.respond_target_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.propose_decline_head = nn.Linear(hidden_size, 1)
        self.respond_decline_head = nn.Linear(hidden_size, 1)
        self.play_head = nn.Linear(hidden_size, 2)
        self.value_head = nn.Linear(hidden_size, 1)
        self._reset_parameters()

    @staticmethod
    def _init_linear(layer: nn.Linear, gain: float) -> None:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    def _reset_parameters(self) -> None:
        tanh_gain = nn.init.calculate_gain("tanh")
        self._init_linear(self.target_encoder[0], tanh_gain)
        self._init_linear(self.target_encoder[2], tanh_gain)
        self._init_linear(self.context_backbone[0], tanh_gain)
        self._init_linear(self.propose_target_head[0], tanh_gain)
        self._init_linear(self.propose_target_head[2], 0.01)
        self._init_linear(self.respond_target_head[0], tanh_gain)
        self._init_linear(self.respond_target_head[2], 0.01)
        self._init_linear(self.propose_decline_head, 0.01)
        self._init_linear(self.respond_decline_head, 0.01)
        self._init_linear(self.play_head, 0.01)
        self._init_linear(self.value_head, 1.0)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        squeeze_batch = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_batch = True

        trust = obs[:, : self.n_agents * 2].reshape(-1, self.n_agents, 2)
        incoming_start = self.n_agents * 2
        incoming_end = incoming_start + self.n_agents
        incoming = obs[:, incoming_start:incoming_end].reshape(-1, self.n_agents, 1)
        reciprocal_trust = obs[:, incoming_end:].reshape(-1, self.n_agents, 2)
        target_inputs = torch.cat([trust, incoming, reciprocal_trust], dim=-1)
        target_features = self.target_encoder(target_inputs)
        context = self.context_backbone(target_features.mean(dim=1))
        expanded_context = context.unsqueeze(1).expand(-1, self.n_agents, -1)
        target_context = torch.cat([target_features, expanded_context], dim=-1)

        propose_target_logits = self.propose_target_head(target_context).squeeze(-1)
        respond_target_logits = self.respond_target_head(target_context).squeeze(-1)
        propose_logits = torch.cat([propose_target_logits, self.propose_decline_head(context)], dim=-1)
        respond_logits = torch.cat([respond_target_logits, self.respond_decline_head(context)], dim=-1)
        play_logits = self.play_head(context)
        value = self.value_head(context).squeeze(-1)

        if squeeze_batch:
            propose_logits = propose_logits.squeeze(0)
            respond_logits = respond_logits.squeeze(0)
            play_logits = play_logits.squeeze(0)
            value = value.squeeze(0)

        return {
            "propose_logits": propose_logits,
            "respond_logits": respond_logits,
            "play_logits": play_logits,
            "value": value,
        }


@dataclass
class RolloutStep:
    obs_propose: torch.Tensor
    action_propose: int
    logprob_propose: float
    proposal_mask: torch.Tensor
    obs_respond: torch.Tensor
    action_respond: int
    logprob_respond: float
    respond_mask: torch.Tensor
    obs_play: torch.Tensor
    action_play: int
    logprob_play: float
    played: bool
    value: float
    reward: float
    done: bool


@dataclass
class PreparedHumanStep:
    timestep: int
    propose_obs: List[torch.Tensor]
    proposals: List[int]
    proposal_masks: List[torch.Tensor]
    logp_prop: List[float]
    value_est: List[float]
    respond_obs: List[torch.Tensor]
    responses: List[int]
    respond_masks: List[torch.Tensor]
    logp_resp: List[float]


def rollout_step_to_dict(step: RolloutStep) -> Dict[str, object]:
    return {
        "obs_propose": step.obs_propose.detach().cpu().clone(),
        "action_propose": int(step.action_propose),
        "logprob_propose": float(step.logprob_propose),
        "proposal_mask": step.proposal_mask.detach().cpu().clone(),
        "obs_respond": step.obs_respond.detach().cpu().clone(),
        "action_respond": int(step.action_respond),
        "logprob_respond": float(step.logprob_respond),
        "respond_mask": step.respond_mask.detach().cpu().clone(),
        "obs_play": step.obs_play.detach().cpu().clone(),
        "action_play": int(step.action_play),
        "logprob_play": float(step.logprob_play),
        "played": bool(step.played),
        "value": float(step.value),
        "reward": float(step.reward),
        "done": bool(step.done),
    }


def rollout_step_from_dict(data: Dict[str, object], device: torch.device) -> RolloutStep:
    return RolloutStep(
        obs_propose=data["obs_propose"].to(device),  # type: ignore[index]
        action_propose=int(data["action_propose"]),
        logprob_propose=float(data["logprob_propose"]),
        proposal_mask=data["proposal_mask"].to(device),  # type: ignore[index]
        obs_respond=data["obs_respond"].to(device),  # type: ignore[index]
        action_respond=int(data["action_respond"]),
        logprob_respond=float(data["logprob_respond"]),
        respond_mask=data["respond_mask"].to(device),  # type: ignore[index]
        obs_play=data["obs_play"].to(device),  # type: ignore[index]
        action_play=int(data["action_play"]),
        logprob_play=float(data["logprob_play"]),
        played=bool(data["played"]),
        value=float(data["value"]),
        reward=float(data["reward"]),
        done=bool(data["done"]),
    )


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: List[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def get_state(self) -> List[Dict[str, object]]:
        return [rollout_step_to_dict(step) for step in self.steps]

    def load_state(self, state: Sequence[Dict[str, object]], device: torch.device) -> None:
        self.steps = [rollout_step_from_dict(step_data, device=device) for step_data in state]


class DecentralizedPPOAgent:
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        n_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.999,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        hidden_size: int = 128,
        max_grad_norm: float = 1.0,
        gae_lambda: float = 0.95,
        normalize_rewards: bool = True,
        normalize_advantages: bool = True,
        device: str = "cpu",
    ) -> None:
        self.agent_id = agent_id
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.device = torch.device(device)

        self.network = AgentNetwork(obs_dim=obs_dim, n_agents=n_agents, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def get_state(self, include_optimizer: bool = True, include_buffer: bool = True) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if include_optimizer else None,
            "buffer_state": self.buffer.get_state() if include_buffer else [],
        }

    def load_state(self, state: Dict[str, object], load_optimizer: bool = True, load_buffer: bool = True) -> None:
        try:
            self.network.load_state_dict(state["network_state"])  # type: ignore[index]
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint weights are incompatible with the current agent network architecture. "
                "Restart training with fresh weights for this version."
            ) from exc
        optimizer_state = state.get("optimizer_state")
        if load_optimizer and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if load_buffer:
            self.buffer.load_state(state.get("buffer_state", []), device=self.device)
        else:
            self.buffer.clear()

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        return masked_logits

    def _sample_action(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, float, float]:
        if mask is not None:
            logits = self._apply_mask(logits, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(dist.entropy().item())

    @torch.no_grad()
    def act_propose(self, obs: torch.Tensor, proposal_mask: torch.Tensor) -> Tuple[int, float, float, float]:
        outputs = self.network(obs)
        action, logp, entropy = self._sample_action(outputs["propose_logits"], proposal_mask)
        value = float(outputs["value"].item())
        return action, logp, entropy, value

    @torch.no_grad()
    def act_respond(self, obs: torch.Tensor, response_mask: torch.Tensor) -> Tuple[int, float, float]:
        outputs = self.network(obs)
        return self._sample_action(outputs["respond_logits"], response_mask)

    @torch.no_grad()
    def act_play(self, obs: torch.Tensor) -> Tuple[int, float, float]:
        outputs = self.network(obs)
        return self._sample_action(outputs["play_logits"], None)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        return float(self.network(obs)["value"].item())

    def _compute_returns_and_advantages(self, bootstrap_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor([step.reward for step in self.buffer.steps], device=self.device, dtype=torch.float32)
        values = torch.tensor([step.value for step in self.buffer.steps], device=self.device, dtype=torch.float32)
        dones = torch.tensor([step.done for step in self.buffer.steps], device=self.device, dtype=torch.float32)

        if self.normalize_rewards:
            if len(rewards) < 2:
                reward_signal = rewards
            else:
                reward_signal = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        else:
            reward_signal = rewards

        advantages = torch.zeros_like(reward_signal)
        next_value = torch.tensor(float(bootstrap_value), device=self.device, dtype=torch.float32)
        gae = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for t in reversed(range(len(reward_signal))):
            non_terminal = 1.0 - dones[t]
            delta = reward_signal[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        if self.normalize_advantages:
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns, advantages

    def update(self, bootstrap_value: float = 0.0) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"loss": 0.0, "actor": 0.0, "critic": 0.0, "entropy": 0.0}

        returns, advantages = self._compute_returns_and_advantages(bootstrap_value=bootstrap_value)
        obs_propose = torch.stack([step.obs_propose for step in self.buffer.steps]).to(self.device)
        obs_respond = torch.stack([step.obs_respond for step in self.buffer.steps]).to(self.device)
        obs_play = torch.stack([step.obs_play for step in self.buffer.steps]).to(self.device)

        actions_propose = torch.tensor(
            [step.action_propose for step in self.buffer.steps], device=self.device, dtype=torch.long
        )
        actions_respond = torch.tensor(
            [step.action_respond for step in self.buffer.steps], device=self.device, dtype=torch.long
        )
        actions_play = torch.tensor(
            [step.action_play if step.played else 0 for step in self.buffer.steps],
            device=self.device,
            dtype=torch.long,
        )
        played_mask = torch.tensor([step.played for step in self.buffer.steps], device=self.device, dtype=torch.bool)

        old_logprob_propose = torch.tensor(
            [step.logprob_propose for step in self.buffer.steps], device=self.device, dtype=torch.float32
        )
        old_logprob_respond = torch.tensor(
            [step.logprob_respond for step in self.buffer.steps], device=self.device, dtype=torch.float32
        )
        old_logprob_play = torch.tensor(
            [step.logprob_play for step in self.buffer.steps], device=self.device, dtype=torch.float32
        )
        old_logprob_total = old_logprob_propose + old_logprob_respond + (old_logprob_play * played_mask.float())

        proposal_masks = torch.stack([step.proposal_mask for step in self.buffer.steps]).to(self.device).bool()
        respond_masks = torch.stack([step.respond_mask for step in self.buffer.steps]).to(self.device).bool()

        total_loss = 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        entropy_mean = 0.0

        for _ in range(self.ppo_epochs):
            out_prop = self.network(obs_propose)
            out_resp = self.network(obs_respond)
            out_play = self.network(obs_play)

            masked_prop_logits = out_prop["propose_logits"].clone()
            masked_prop_logits[~proposal_masks] = -1e9
            prop_dist = Categorical(logits=masked_prop_logits)
            prop_logp = prop_dist.log_prob(actions_propose)
            prop_entropy = prop_dist.entropy()

            masked_resp_logits = out_resp["respond_logits"].clone()
            masked_resp_logits[~respond_masks] = -1e9
            resp_dist = Categorical(logits=masked_resp_logits)
            resp_logp = resp_dist.log_prob(actions_respond)
            resp_entropy = resp_dist.entropy()

            play_dist = Categorical(logits=out_play["play_logits"])
            play_logp = play_dist.log_prob(actions_play)
            play_entropy = play_dist.entropy() * played_mask.float()

            new_logprob_total = prop_logp + resp_logp + (play_logp * played_mask.float())
            entropy = prop_entropy + resp_entropy + play_entropy

            ratio = torch.exp(new_logprob_total - old_logprob_total)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            objective = torch.min(ratio * advantages, clipped_ratio * advantages)
            l_clip = objective.mean()

            values = out_prop["value"]
            l_vf = nn.functional.mse_loss(values, returns)
            entropy_bonus = entropy.mean()

            loss = -l_clip + self.value_coef * l_vf - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            total_loss += float(loss.item())
            actor_loss += float((-l_clip).item())
            critic_loss += float(l_vf.item())
            entropy_mean += float(entropy_bonus.item())

        self.buffer.clear()

        epochs = float(self.ppo_epochs)
        return {
            "loss": total_loss / epochs,
            "actor": actor_loss / epochs,
            "critic": critic_loss / epochs,
            "entropy": entropy_mean / epochs,
        }


@dataclass
class EpisodeTracker:
    episode_returns: List[float]
    total_pairs: int
    step_history: List[Dict[str, object]]
    done: bool = False
    step_count: int = 0
    loss_sum: float = 0.0
    actor_sum: float = 0.0
    critic_sum: float = 0.0
    entropy_sum: float = 0.0
    update_count: int = 0


def step_history_entry_to_cpu(entry: Dict[str, object]) -> Dict[str, object]:
    out = dict(entry)
    out["reputation"] = entry["reputation"].detach().cpu().clone()  # type: ignore[index]
    out["confidence"] = entry["confidence"].detach().cpu().clone()  # type: ignore[index]
    out["pairs"] = list(entry["pairs"])  # type: ignore[index]
    out["proposals"] = list(entry["proposals"])  # type: ignore[index]
    out["responses"] = list(entry["responses"])  # type: ignore[index]
    out["play_actions"] = dict(entry["play_actions"])  # type: ignore[index]
    out["interaction_outcomes"] = [dict(item) for item in entry["interaction_outcomes"]]  # type: ignore[index]
    out["agent_rewards"] = [float(v) for v in entry["agent_rewards"]]  # type: ignore[index]
    out["societal_reward"] = float(entry["societal_reward"])
    out["befriend_count"] = int(entry["befriend_count"])
    out["betray_count"] = int(entry["betray_count"])
    return out


def serialize_episode_tracker(tracker: EpisodeTracker) -> Dict[str, object]:
    return {
        "episode_returns": [float(v) for v in tracker.episode_returns],
        "total_pairs": int(tracker.total_pairs),
        "step_history": [step_history_entry_to_cpu(entry) for entry in tracker.step_history],
        "done": bool(tracker.done),
        "step_count": int(tracker.step_count),
        "loss_sum": float(tracker.loss_sum),
        "actor_sum": float(tracker.actor_sum),
        "critic_sum": float(tracker.critic_sum),
        "entropy_sum": float(tracker.entropy_sum),
        "update_count": int(tracker.update_count),
    }


def deserialize_episode_tracker(data: Dict[str, object]) -> EpisodeTracker:
    return EpisodeTracker(
        episode_returns=[float(v) for v in data["episode_returns"]],  # type: ignore[index]
        total_pairs=int(data["total_pairs"]),
        step_history=[step_history_entry_to_cpu(entry) for entry in data["step_history"]],  # type: ignore[index]
        done=bool(data["done"]),
        step_count=int(data.get("step_count", 0)),
        loss_sum=float(data.get("loss_sum", 0.0)),
        actor_sum=float(data.get("actor_sum", 0.0)),
        critic_sum=float(data.get("critic_sum", 0.0)),
        entropy_sum=float(data.get("entropy_sum", 0.0)),
        update_count=int(data.get("update_count", 0)),
    )


class MiniSocietyTrainer:
    def __init__(
        self,
        n_agents: int = 5,
        episode_length: int = 64,
        confidence_decay: float = 0.005,
        gossip_noise: float = 0.04,
        gossip_strength: float = 0.18,
        initial_confidence: float = 0.15,
        reputation_delta_befriend: float = 0.15,
        reputation_delta_betray: float = -0.35,
        confidence_gain_interaction: float = 1.0,
        payoff: Optional[PayoffConfig] = None,
        gamma: float = 0.999,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        hidden_size: int = 128,
        max_grad_norm: float = 1.0,
        gae_lambda: float = 0.95,
        rollout_steps: int = 128,
        normalize_rewards: bool = True,
        normalize_advantages: bool = True,
        human_agent_id: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        if human_agent_id is not None and (human_agent_id < 0 or human_agent_id >= n_agents):
            raise ValueError(f"human_agent_id must be in [0, {n_agents - 1}] or None.")
        resolved_payoff = payoff or PayoffConfig()
        self.config: Dict[str, object] = {
            "n_agents": n_agents,
            "episode_length": episode_length,
            "confidence_decay": confidence_decay,
            "gossip_noise": gossip_noise,
            "gossip_strength": gossip_strength,
            "initial_confidence": initial_confidence,
            "reputation_delta_befriend": reputation_delta_befriend,
            "reputation_delta_betray": reputation_delta_betray,
            "confidence_gain_interaction": confidence_gain_interaction,
            "payoff": {
                "mutual_befriend": tuple(resolved_payoff.mutual_befriend),
                "betray_befriend": tuple(resolved_payoff.betray_befriend),
                "befriend_betray": tuple(resolved_payoff.befriend_betray),
                "mutual_betray": tuple(resolved_payoff.mutual_betray),
                "isolation": float(resolved_payoff.isolation),
            },
            "gamma": gamma,
            "lr": lr,
            "clip_epsilon": clip_epsilon,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "ppo_epochs": ppo_epochs,
            "hidden_size": hidden_size,
            "max_grad_norm": max_grad_norm,
            "gae_lambda": gae_lambda,
            "rollout_steps": rollout_steps,
            "normalize_rewards": normalize_rewards,
            "normalize_advantages": normalize_advantages,
            "human_agent_id": human_agent_id,
            "device": device,
        }
        self.env = MiniSocietyEnv(
            n_agents=n_agents,
            episode_length=episode_length,
            confidence_decay=confidence_decay,
            gossip_noise=gossip_noise,
            gossip_strength=gossip_strength,
            initial_confidence=initial_confidence,
            reputation_delta_befriend=reputation_delta_befriend,
            reputation_delta_betray=reputation_delta_betray,
            confidence_gain_interaction=confidence_gain_interaction,
            payoff=resolved_payoff,
            device=device,
        )
        self.n_agents = n_agents
        self.device = torch.device(device)
        self.rollout_steps = rollout_steps
        self.human_agent_id = human_agent_id
        self.agents = [
            DecentralizedPPOAgent(
                agent_id=i,
                obs_dim=self.env.obs_dim,
                n_agents=n_agents,
                gamma=gamma,
                lr=lr,
                clip_epsilon=clip_epsilon,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                ppo_epochs=ppo_epochs,
                hidden_size=hidden_size,
                max_grad_norm=max_grad_norm,
                gae_lambda=gae_lambda,
                normalize_rewards=normalize_rewards,
                normalize_advantages=normalize_advantages,
                device=device,
            )
            for i in range(n_agents)
        ]

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "MiniSocietyTrainer":
        payoff_data = config["payoff"]
        payoff = PayoffConfig(
            mutual_befriend=tuple(payoff_data["mutual_befriend"]),  # type: ignore[index]
            betray_befriend=tuple(payoff_data["betray_befriend"]),  # type: ignore[index]
            befriend_betray=tuple(payoff_data["befriend_betray"]),  # type: ignore[index]
            mutual_betray=tuple(payoff_data["mutual_betray"]),  # type: ignore[index]
            isolation=float(payoff_data["isolation"]),  # type: ignore[index]
        )
        return cls(
            n_agents=int(config["n_agents"]),
            episode_length=int(config["episode_length"]),
            confidence_decay=float(config["confidence_decay"]),
            gossip_noise=float(config["gossip_noise"]),
            gossip_strength=float(config.get("gossip_strength", 0.18)),
            initial_confidence=float(config["initial_confidence"]),
            reputation_delta_befriend=float(config["reputation_delta_befriend"]),
            reputation_delta_betray=float(config["reputation_delta_betray"]),
            confidence_gain_interaction=float(config["confidence_gain_interaction"]),
            payoff=payoff,
            gamma=float(config["gamma"]),
            lr=float(config["lr"]),
            clip_epsilon=float(config["clip_epsilon"]),
            value_coef=float(config["value_coef"]),
            entropy_coef=float(config["entropy_coef"]),
            ppo_epochs=int(config["ppo_epochs"]),
            hidden_size=int(config["hidden_size"]),
            max_grad_norm=float(config["max_grad_norm"]),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            rollout_steps=int(config.get("rollout_steps", 128)),
            normalize_rewards=bool(config["normalize_rewards"]),
            normalize_advantages=bool(config["normalize_advantages"]),
            human_agent_id=(
                None
                if config.get("human_agent_id") is None
                else int(config["human_agent_id"])
            ),
            device=str(config["device"]),
        )

    @staticmethod
    def _build_step_entry(info: Dict[str, object], rewards: Sequence[float]) -> Dict[str, object]:
        return {
            "pairs": list(info["pairs"]),
            "proposals": list(info["proposals"]),
            "responses": list(info["responses"]),
            "play_actions": dict(info["play_actions"]),
            "interaction_outcomes": list(info["interaction_outcomes"]),
            "reputation": info["reputation"].detach().cpu().clone(),
            "confidence": info["confidence"].detach().cpu().clone(),
            "societal_reward": float(info["societal_reward"]),
            "befriend_count": int(info["befriend_count"]),
            "betray_count": int(info["betray_count"]),
            "agent_rewards": [float(r) for r in rewards],
        }

    def start_episode(self, collect_history: bool = False) -> EpisodeTracker:
        self.env.reset()
        return EpisodeTracker(
            episode_returns=[0.0 for _ in range(self.n_agents)],
            total_pairs=0,
            step_history=[],
            done=False,
        )

    def is_human_agent(self, agent_id: int) -> bool:
        return self.human_agent_id is not None and agent_id == self.human_agent_id

    def _trainable_agent_ids(self) -> List[int]:
        return [i for i in range(self.n_agents) if not self.is_human_agent(i)]

    def _mean_stats(self, stats: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if len(stats) == 0:
            return {"loss": 0.0, "actor": 0.0, "critic": 0.0, "entropy": 0.0}
        return {
            "loss": sum(s["loss"] for s in stats) / len(stats),
            "actor": sum(s["actor"] for s in stats) / len(stats),
            "critic": sum(s["critic"] for s in stats) / len(stats),
            "entropy": sum(s["entropy"] for s in stats) / len(stats),
        }

    def _accumulate_stats(self, tracker: EpisodeTracker, stats: Dict[str, float]) -> None:
        tracker.loss_sum += stats["loss"]
        tracker.actor_sum += stats["actor"]
        tracker.critic_sum += stats["critic"]
        tracker.entropy_sum += stats["entropy"]
        tracker.update_count += 1

    def _current_bootstrap_values(self, agent_ids: Sequence[int]) -> Dict[int, float]:
        values: Dict[int, float] = {}
        for i in agent_ids:
            agent = self.agents[i]
            obs = self.env.observe(i).to(self.device)
            values[i] = agent.value(obs)
        return values

    def _maybe_update_agents(self, force: bool, terminal: bool) -> Optional[Dict[str, float]]:
        trainable_ids = self._trainable_agent_ids()
        if len(trainable_ids) == 0:
            return None
        min_buffer_len = min(len(self.agents[i].buffer) for i in trainable_ids)
        if not force and min_buffer_len < self.rollout_steps:
            return None
        if force and min_buffer_len == 0:
            return None

        bootstrap_values = (
            {i: 0.0 for i in trainable_ids}
            if terminal
            else self._current_bootstrap_values(trainable_ids)
        )
        stats = [
            self.agents[i].update(bootstrap_value=bootstrap_values[i])
            for i in trainable_ids
        ]
        return self._mean_stats(stats)

    def _record_step(
        self,
        tracker: EpisodeTracker,
        rewards: Sequence[float],
        done: bool,
        info: Dict[str, object],
        rollout_steps: Sequence[RolloutStep],
        collect_history: bool,
    ) -> Dict[str, object]:
        tracker.total_pairs += len(info["pairs"])  # type: ignore[index]
        tracker.step_count += 1
        info_out = dict(info)
        info_out["agent_rewards"] = [float(v) for v in rewards]

        for i in range(self.n_agents):
            tracker.episode_returns[i] += rewards[i]
            if not self.is_human_agent(i):
                self.agents[i].buffer.add(rollout_steps[i])

        if collect_history:
            tracker.step_history.append(self._build_step_entry(info=info_out, rewards=rewards))

        # Non-terminal periodic PPO updates (rolling horizon) so agents can train without fixed episode end.
        if not done:
            periodic_stats = self._maybe_update_agents(force=False, terminal=False)
            if periodic_stats is not None:
                self._accumulate_stats(tracker, periodic_stats)

        tracker.done = done
        return info_out

    def prepare_human_step(self, human_propose_action: int) -> PreparedHumanStep:
        if self.human_agent_id is None:
            raise RuntimeError("prepare_human_step called without a configured human agent.")
        human_id = self.human_agent_id
        if self.env.episode_length > 0 and self.env.timestep >= self.env.episode_length:
            raise RuntimeError("Cannot prepare step: environment episode is already finished.")

        propose_obs = [self.env.observe(i).to(self.device) for i in range(self.n_agents)]
        proposals: List[int] = []
        logp_prop: List[float] = []
        proposal_masks: List[torch.Tensor] = []
        value_est: List[float] = []

        for i in range(self.n_agents):
            mask = self.env.proposal_mask(i)
            if i == human_id:
                action = int(human_propose_action)
                if action < 0 or action > self.n_agents or not bool(mask[action]):
                    raise ValueError(
                        f"Invalid human propose action={action}. "
                        f"Expected target in [0,{self.n_agents}] excluding self ({human_id})."
                    )
                logp = 0.0
                value = 0.0
            else:
                action, logp, _, value = self.agents[i].act_propose(propose_obs[i], mask)
            proposals.append(action)
            logp_prop.append(logp)
            proposal_masks.append(mask.detach().clone())
            value_est.append(float(value))

        self.env._build_incoming_proposals(proposals)

        respond_obs = [self.env.observe(i).to(self.device) for i in range(self.n_agents)]
        responses: List[int] = []
        respond_masks: List[torch.Tensor] = []
        logp_resp: List[float] = []
        for i in range(self.n_agents):
            mask = self.env.response_mask(i)
            if i == human_id:
                action = self.n_agents  # pending human decision
                logp = 0.0
            else:
                action, logp, _ = self.agents[i].act_respond(respond_obs[i], mask)
            responses.append(int(action))
            logp_resp.append(float(logp))
            respond_masks.append(mask.detach().clone())

        return PreparedHumanStep(
            timestep=self.env.timestep,
            propose_obs=[obs.detach().clone() for obs in propose_obs],
            proposals=proposals,
            proposal_masks=proposal_masks,
            logp_prop=logp_prop,
            value_est=value_est,
            respond_obs=[obs.detach().clone() for obs in respond_obs],
            responses=responses,
            respond_masks=respond_masks,
            logp_resp=logp_resp,
        )

    def execute_prepared_human_step(
        self,
        tracker: EpisodeTracker,
        prepared: PreparedHumanStep,
        human_response_action: int,
        human_play_action: int,
        collect_history: bool = False,
    ) -> Dict[str, object]:
        if self.human_agent_id is None:
            raise RuntimeError("execute_prepared_human_step called without a configured human agent.")
        human_id = self.human_agent_id
        if prepared.timestep != self.env.timestep:
            raise RuntimeError("Prepared human step is stale (environment timestep changed). Prepare a new step.")

        responses = list(prepared.responses)
        human_response = int(human_response_action)
        human_mask = prepared.respond_masks[human_id]
        if human_response < 0 or human_response > self.n_agents or not bool(human_mask[human_response]):
            human_response = self.n_agents  # decline all if invalid
        responses[human_id] = human_response

        pairs = self.env._form_pairs(prepared.proposals, responses)
        self.env._set_play_context_incoming(pairs)
        paired_agents = {agent for pair in pairs for agent in pair}
        play_actions: Dict[int, int] = {}
        play_obs: List[torch.Tensor] = [
            torch.zeros(self.env.obs_dim, dtype=torch.float32, device=self.device)
            for _ in range(self.n_agents)
        ]
        play_action: List[int] = [-1 for _ in range(self.n_agents)]
        play_logp: List[float] = [0.0 for _ in range(self.n_agents)]
        played: List[bool] = [False for _ in range(self.n_agents)]

        for agent_id in paired_agents:
            obs = self.env.observe(agent_id).to(self.device)
            if agent_id == human_id:
                action = 1 if int(human_play_action) > 0 else 0
                logp = 0.0
            else:
                action, logp, _ = self.agents[agent_id].act_play(obs)
            play_actions[agent_id] = int(action)
            play_obs[agent_id] = obs.detach().clone()
            play_action[agent_id] = int(action)
            play_logp[agent_id] = float(logp)
            played[agent_id] = True

        _, rewards, done, info = self.env._resolve_step(
            pairs=pairs,
            proposals=prepared.proposals,
            responses=responses,
            play_actions=play_actions,
        )

        rollout_steps: List[RolloutStep] = []
        for i in range(self.n_agents):
            rollout_steps.append(
                RolloutStep(
                    obs_propose=prepared.propose_obs[i].detach().clone(),
                    action_propose=int(prepared.proposals[i]),
                    logprob_propose=float(prepared.logp_prop[i]),
                    proposal_mask=prepared.proposal_masks[i].detach().clone(),
                    obs_respond=prepared.respond_obs[i].detach().clone(),
                    action_respond=int(responses[i]),
                    logprob_respond=float(prepared.logp_resp[i]),
                    respond_mask=prepared.respond_masks[i].detach().clone(),
                    obs_play=play_obs[i].detach().clone(),
                    action_play=int(play_action[i]),
                    logprob_play=float(play_logp[i]),
                    played=bool(played[i]),
                    value=float(prepared.value_est[i]),
                    reward=float(rewards[i]),
                    done=bool(done),
                )
            )
        return self._record_step(
            tracker=tracker,
            rewards=rewards,
            done=done,
            info=info,
            rollout_steps=rollout_steps,
            collect_history=collect_history,
        )

    def step_episode(
        self,
        tracker: EpisodeTracker,
        collect_history: bool = False,
        human_actions: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, object]:
        if self.human_agent_id is None:
            _, rewards, done, info, rollout_steps = self.env.step(self.agents)
            return self._record_step(
                tracker=tracker,
                rewards=rewards,
                done=done,
                info=info,
                rollout_steps=rollout_steps,
                collect_history=collect_history,
            )

        if human_actions is None:
            raise RuntimeError(
                "human_actions must be provided when a human-controlled agent is enabled."
            )
        human_propose, human_response, human_play = human_actions
        prepared = self.prepare_human_step(human_propose_action=int(human_propose))
        return self.execute_prepared_human_step(
            tracker=tracker,
            prepared=prepared,
            human_response_action=int(human_response),
            human_play_action=int(human_play),
            collect_history=collect_history,
        )

    def finish_episode(self, tracker: EpisodeTracker, terminal: bool = True) -> Dict[str, object]:
        final_stats = self._maybe_update_agents(force=True, terminal=terminal)
        if final_stats is not None:
            self._accumulate_stats(tracker, final_stats)

        if tracker.update_count > 0:
            mean_loss = tracker.loss_sum / tracker.update_count
            mean_entropy = tracker.entropy_sum / tracker.update_count
        else:
            mean_loss = 0.0
            mean_entropy = 0.0
        mean_return = sum(tracker.episode_returns) / self.n_agents

        result: Dict[str, object] = {
            "episode_returns": tracker.episode_returns,
            "mean_return": mean_return,
            "mean_loss": mean_loss,
            "mean_entropy": mean_entropy,
            "pair_count": tracker.total_pairs,
            "updates": tracker.update_count,
            "steps": tracker.step_count,
            "terminal": bool(terminal),
        }
        if len(tracker.step_history) > 0:
            result["step_history"] = tracker.step_history
        return result

    def run_episode(self, collect_history: bool = False) -> Dict[str, object]:
        if self.human_agent_id is not None:
            raise RuntimeError("run_episode is not available in human-player mode; use interactive stepping.")
        tracker = self.start_episode(collect_history=collect_history)
        while not tracker.done:
            self.step_episode(tracker, collect_history=collect_history)
        return self.finish_episode(tracker)

    def get_state(self, include_optimizer: bool = True, include_buffers: bool = True) -> Dict[str, object]:
        return {
            "config": dict(self.config),
            "env_state": self.env.get_state(),
            "agent_states": [
                agent.get_state(include_optimizer=include_optimizer, include_buffer=include_buffers)
                for agent in self.agents
            ],
        }

    def load_state(
        self,
        state: Dict[str, object],
        load_optimizer: bool = True,
        load_buffers: bool = True,
        load_env: bool = True,
    ) -> None:
        if load_env:
            self.env.load_state(state["env_state"])  # type: ignore[index]
        agent_states = state["agent_states"]  # type: ignore[index]
        if len(agent_states) != len(self.agents):
            raise ValueError(
                f"Checkpoint has {len(agent_states)} agents, current trainer has {len(self.agents)} agents."
            )
        for agent, agent_state in zip(self.agents, agent_states):
            agent.load_state(agent_state, load_optimizer=load_optimizer, load_buffer=load_buffers)


class MiniSocietyVisualizer:
    def __init__(
        self,
        n_agents: int,
        human_agent_id: Optional[int] = None,
        cooperation_window: int = 10,
        reward_window: int = 10,
        pause_seconds: float = 0.25,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            from matplotlib.lines import Line2D
            from matplotlib.patches import FancyArrowPatch
        except ImportError as exc:
            raise RuntimeError("Visualization requires matplotlib. Install with: pip install matplotlib") from exc

        self.n_agents = n_agents
        self.human_agent_id = human_agent_id
        self.cooperation_window = cooperation_window
        self.reward_window = reward_window
        self.pause_seconds = pause_seconds

        self.plt = plt
        self.Line2D = Line2D
        self.FancyArrowPatch = FancyArrowPatch
        self.trust_cmap = LinearSegmentedColormap.from_list(
            "trust_cmap",
            ["#b2182b", "#ffffff", "#1a9850"],
        )

        self.fig = None
        self.ax_graph = None
        self.ax_heat = None
        self.ax_metrics = None
        self.ax_metrics_right = None
        self.heat_colorbar = None
        self._live_interactive_backend = None

        self.positions: Dict[int, Tuple[float, float]] = {}
        for i in range(n_agents):
            angle = 2.0 * math.pi * (i / max(1, n_agents))
            self.positions[i] = (math.cos(angle), math.sin(angle))

    def _ensure_figure(self) -> None:
        if self.fig is not None:
            return
        self.fig, axes = self.plt.subplots(
            1,
            3,
            figsize=(18, 6.8),
            gridspec_kw={"width_ratios": [1.05, 1.15, 1.2]},
        )
        self.ax_graph, self.ax_heat, self.ax_metrics = axes
        self.ax_metrics_right = self.ax_metrics.twinx()
        self.fig.subplots_adjust(left=0.045, right=0.94, bottom=0.18, top=0.82, wspace=0.42)

    def _apply_layout(self, title: str) -> None:
        assert self.fig is not None
        self.fig.subplots_adjust(left=0.045, right=0.94, bottom=0.18, top=0.82, wspace=0.42)
        self.fig.suptitle(title, fontsize=14, y=0.955)

    @staticmethod
    def _rolling_mean(values: Sequence[float], window: int) -> List[float]:
        if window <= 1:
            return [float(v) for v in values]
        out: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            out.append(float(sum(chunk) / len(chunk)))
        return out

    @staticmethod
    def _rolling_cooperation_rate(history: Sequence[Dict[str, object]], window: int) -> List[float]:
        rates: List[float] = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            befriend = 0
            betray = 0
            for j in range(start, i + 1):
                befriend += int(history[j]["befriend_count"])
                betray += int(history[j]["betray_count"])
            total_actions = befriend + betray
            rate = 100.0 * befriend / total_actions if total_actions > 0 else 0.0
            rates.append(rate)
        return rates

    def _draw_edge(
        self,
        ax: object,
        source: int,
        target: int,
        color: str,
        linewidth: float,
        linestyle: str,
        alpha: float,
        rad: float,
    ) -> None:
        if source == target:
            return
        x1, y1 = self.positions[source]
        x2, y2 = self.positions[target]
        patch = self.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13.0 + linewidth * 1.5,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            shrinkA=22,
            shrinkB=22,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(patch)

    def _draw_social_graph(self, step_data: Dict[str, object], step_idx: int, total_steps: int) -> None:
        assert self.ax_graph is not None
        ax = self.ax_graph
        ax.clear()
        ax.set_title(f"Panel 1: Social Graph (Step {step_idx}/{total_steps})")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")

        proposals = list(step_data["proposals"])
        for proposer, target_obj in enumerate(proposals):
            target = int(target_obj)
            if target < self.n_agents and target != proposer:
                self._draw_edge(
                    ax=ax,
                    source=proposer,
                    target=target,
                    color="#9e9e9e",
                    linewidth=1.2,
                    linestyle="dashed",
                    alpha=0.75,
                    rad=0.08,
                )

        outcomes = list(step_data["interaction_outcomes"])
        for outcome_item in outcomes:
            agents = outcome_item["agents"]
            actions = outcome_item["actions"]
            agent_a = int(agents[0])
            agent_b = int(agents[1])
            action_a = int(actions[0])
            action_b = int(actions[1])

            if action_a == 1 and action_b == 1:
                color_ab = "#2ca02c"
                color_ba = "#2ca02c"
            elif action_a == 0 and action_b == 0:
                color_ab = "#d62728"
                color_ba = "#d62728"
            elif action_a == 0 and action_b == 1:
                color_ab = "#ff7f0e"  # betrayer
                color_ba = "#9467bd"  # befriender
            else:
                color_ab = "#9467bd"
                color_ba = "#ff7f0e"

            self._draw_edge(
                ax=ax,
                source=agent_a,
                target=agent_b,
                color=color_ab,
                linewidth=3.0,
                linestyle="solid",
                alpha=0.95,
                rad=0.23,
            )
            self._draw_edge(
                ax=ax,
                source=agent_b,
                target=agent_a,
                color=color_ba,
                linewidth=3.0,
                linestyle="solid",
                alpha=0.95,
                rad=-0.23,
            )

        for i in range(self.n_agents):
            x, y = self.positions[i]
            ax.scatter(
                [x],
                [y],
                s=820,
                c="#f8f8f8",
                edgecolors="#1f1f1f",
                linewidths=1.6,
                zorder=3,
            )
            label = f"H{i}" if self.human_agent_id is not None and i == self.human_agent_id else f"A{i}"
            ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

        legend_handles = [
            self.Line2D([0], [0], color="#9e9e9e", lw=1.5, linestyle="dashed", label="Proposal"),
            self.Line2D([0], [0], color="#2ca02c", lw=3.0, label="Mutual Befriend"),
            self.Line2D([0], [0], color="#d62728", lw=3.0, label="Mutual Betray"),
            self.Line2D([0], [0], color="#ff7f0e", lw=3.0, label="Asym: Betrayer"),
            self.Line2D([0], [0], color="#9467bd", lw=3.0, label="Asym: Befriender"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            fontsize=8,
            ncol=2,
        )

    def _draw_trust_heatmap(self, step_data: Dict[str, object]) -> None:
        assert self.ax_heat is not None
        assert self.fig is not None
        ax = self.ax_heat
        ax.clear()

        reputation_t = step_data["reputation"]
        confidence_t = step_data["confidence"]
        reputation = reputation_t.numpy()
        confidence = confidence_t.numpy().clip(0.0, 1.0)

        heat = ax.imshow(reputation, cmap=self.trust_cmap, vmin=-1.0, vmax=1.0, alpha=confidence)
        ax.set_title("Panel 2: Trust Matrix (Reputation, Alpha=Confidence)")
        ax.set_xlabel("Target Agent", labelpad=6)
        ax.set_ylabel("Observing Agent", labelpad=6)
        labels = [f"A{i}" for i in range(self.n_agents)]
        ax.set_xticks(range(self.n_agents), labels=labels, rotation=30, ha="right")
        ax.set_yticks(range(self.n_agents), labels=labels)
        ax.tick_params(axis="both", labelsize=9)

        if self.heat_colorbar is None:
            self.heat_colorbar = self.fig.colorbar(heat, ax=ax, fraction=0.048, pad=0.04)
            self.heat_colorbar.set_label("Reputation")
        else:
            self.heat_colorbar.update_normal(heat)

    def _draw_metrics(self, history: Sequence[Dict[str, object]]) -> None:
        assert self.ax_metrics is not None
        assert self.ax_metrics_right is not None
        ax_l = self.ax_metrics
        ax_r = self.ax_metrics_right
        ax_l.clear()
        ax_r.clear()

        rewards = [float(step["societal_reward"]) for step in history]
        reward_smoothed = self._rolling_mean(rewards, self.reward_window)
        coop_rate = self._rolling_cooperation_rate(history, self.cooperation_window)
        steps = list(range(1, len(history) + 1))

        ax_l.plot(steps, reward_smoothed, color="#1f77b4", linewidth=2.0, label=f"Reward MA({self.reward_window})")
        ax_l.scatter(steps, rewards, color="#1f77b4", s=10, alpha=0.18)
        ax_l.axhline(y=0.0, color="#bbbbbb", linewidth=1.0)
        ax_l.set_xlabel("Time Step")
        ax_l.set_ylabel("Total Societal Reward", labelpad=8)
        ax_l.yaxis.set_label_position("left")
        ax_l.yaxis.tick_left()
        ax_l.set_title("Panel 3: Global Metrics")

        ax_r.plot(
            steps,
            coop_rate,
            color="#ff7f0e",
            linewidth=2.0,
            label=f"Cooperation Rate K={self.cooperation_window}",
        )
        ax_r.set_ylabel("Cooperation Rate (%)", labelpad=10)
        ax_r.set_ylim(0.0, 100.0)
        ax_r.spines["right"].set_position(("axes", 1.0))
        ax_r.spines["left"].set_visible(False)
        ax_r.yaxis.set_label_position("right")
        ax_r.yaxis.tick_right()
        ax_r.yaxis.set_label_coords(1.12, 0.5)
        ax_l.tick_params(axis="both", labelsize=9)
        ax_r.tick_params(axis="y", labelsize=9)
        ax_l.grid(True, alpha=0.18, linewidth=0.6)

        lines_l, labels_l = ax_l.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", frameon=False, fontsize=8)

    def animate_episode(self, step_history: Sequence[Dict[str, object]], output_path: Optional[str] = None) -> None:
        if len(step_history) == 0:
            return
        self._ensure_figure()
        assert self.fig is not None

        backend = str(self.plt.get_backend()).lower()
        interactive_backend = "agg" not in backend
        if interactive_backend:
            self.plt.ion()
        drawn_history: List[Dict[str, object]] = []
        total_steps = len(step_history)

        for step_idx, step_data in enumerate(step_history, start=1):
            drawn_history.append(step_data)
            self._draw_social_graph(step_data, step_idx, total_steps)
            self._draw_trust_heatmap(step_data)
            self._draw_metrics(drawn_history)
            self._apply_layout("Mini-Society Visualization")
            if interactive_backend:
                self.plt.pause(self.pause_seconds)

        if output_path:
            self.fig.savefig(output_path, dpi=170, bbox_inches="tight")

        if interactive_backend:
            self.plt.ioff()
            self.plt.show()

    def render_step(self, history: Sequence[Dict[str, object]]) -> None:
        if len(history) == 0:
            return
        self._ensure_figure()
        assert self.fig is not None

        if self._live_interactive_backend is None:
            backend = str(self.plt.get_backend()).lower()
            self._live_interactive_backend = "agg" not in backend
            if self._live_interactive_backend:
                self.plt.ion()

        step_data = history[-1]
        step_idx = len(history)
        self._draw_social_graph(step_data, step_idx=step_idx, total_steps=max(step_idx, 1))
        self._draw_trust_heatmap(step_data)
        self._draw_metrics(history)
        self._apply_layout("Mini-Society Visualization (Live)")
        if self._live_interactive_backend:
            self.plt.pause(max(self.pause_seconds, 1e-3))

    def finalize_live(self, output_path: Optional[str] = None) -> None:
        if self.fig is None:
            return
        if output_path:
            self.fig.savefig(output_path, dpi=170, bbox_inches="tight")
        if self._live_interactive_backend:
            self.plt.ioff()
            self.plt.show()


class ConsolePauseWatcher:
    """Windows-friendly non-blocking key polling for pause requests."""

    def __init__(self) -> None:
        self.enabled = os.name == "nt"
        self._msvcrt = None
        if self.enabled:
            try:
                import msvcrt  # type: ignore

                self._msvcrt = msvcrt
            except ImportError:
                self.enabled = False

    def poll_pause_requested(self) -> bool:
        if not self.enabled or self._msvcrt is None:
            return False
        requested = False
        while self._msvcrt.kbhit():
            char = self._msvcrt.getwch().lower()
            if char == "p":
                requested = True
        return requested


def _serialize_rng_state() -> Dict[str, object]:
    return {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python_random_state"])  # type: ignore[index]
    torch.set_rng_state(state["torch_rng_state"])  # type: ignore[index]
    cuda_state = state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: str,
    trainer: MiniSocietyTrainer,
    current_episode: int,
    total_episodes: int,
    tracker: Optional[EpisodeTracker],
    global_steps: int = 0,
    global_scores: Optional[Sequence[float]] = None,
) -> None:
    payload: Dict[str, object] = {
        "version": 1,
        "trainer_state": trainer.get_state(include_optimizer=True, include_buffers=True),
        "run_state": {
            "current_episode": int(current_episode),
            "total_episodes": int(total_episodes),
            "tracker": None if tracker is None else serialize_episode_tracker(tracker),
            "global_steps": int(global_steps),
            "global_scores": None if global_scores is None else [float(v) for v in global_scores],
        },
        "rng_state": _serialize_rng_state(),
    }
    checkpoint_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str, device: str) -> Dict[str, object]:
    # weights_only=False is required for non-tensor python objects in the checkpoint payload.
    checkpoint_path = os.path.abspath(path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def horizon_text(episode_length: int) -> str:
    return "inf" if episode_length <= 0 else str(episode_length)


def format_gui_status(
    *,
    phase: str,
    mode_text: str,
    episode_text: str,
    step_now: int,
    episode_length: int,
    global_steps: int,
    pending_text: str,
) -> str:
    return (
        f"phase={phase}  mode={mode_text}  episode={episode_text}\n"
        f"step={step_now}/{horizon_text(episode_length)}  "
        f"global_steps={global_steps}  pending_human_step={pending_text}"
    )


def format_gui_leaderboard(
    global_scores: Sequence[float],
    *,
    human_mode: bool,
    human_agent_id: Optional[int],
    entries_per_row: int = 3,
) -> str:
    rows: List[Tuple[str, float]] = []
    for idx, score in enumerate(global_scores):
        if human_mode and human_agent_id == idx:
            label = f"Human (A{idx})"
        else:
            label = f"AI A{idx}"
        rows.append((label, float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)

    lines = ["Leaderboard:"]
    for start in range(0, len(rows), max(1, entries_per_row)):
        chunk = rows[start : start + max(1, entries_per_row)]
        lines.append(" | ".join(f"{name}: {score:+8.2f}" for name, score in chunk))
    return "\n".join(lines)


def format_agent_state_snapshot(trainer: MiniSocietyTrainer) -> str:
    lines: List[str] = [f"Environment step={trainer.env.timestep}/{horizon_text(trainer.env.episode_length)}"]
    for observer_id, state in enumerate(trainer.env.states):
        rep = ", ".join(f"{v:+.2f}" for v in state.trust_matrix[:, 0].detach().cpu().tolist())
        conf = ", ".join(f"{v:.2f}" for v in state.trust_matrix[:, 1].detach().cpu().tolist())
        incoming = ", ".join(str(int(v)) for v in state.incoming_proposals.detach().cpu().tolist())
        role = " (Human)" if trainer.is_human_agent(observer_id) else ""
        lines.append(f"Agent A{observer_id}{role}")
        lines.append(f"  reputation: [{rep}]")
        lines.append(f"  confidence: [{conf}]")
        lines.append(f"  incoming proposals: [{incoming}]")
        lines.append(f"  replay buffer length: {len(trainer.agents[observer_id].buffer)}")
    return "\n".join(lines)


def print_agent_state_snapshot(trainer: MiniSocietyTrainer) -> None:
    print(f"\n{format_agent_state_snapshot(trainer)}")


def format_recent_interactions(history: Sequence[Dict[str, object]], tail: int = 5) -> str:
    if len(history) == 0:
        return "No interaction history yet for current episode."
    start = max(0, len(history) - tail)
    lines: List[str] = [f"Recent interactions (last {len(history) - start} of {len(history)} steps):"]
    for idx in range(start, len(history)):
        step = history[idx]
        outcomes = step["interaction_outcomes"]
        lines.append(
            f"step={idx + 1:03d} societal_reward={float(step['societal_reward']):+7.2f} "
            f"pairs={len(step['pairs'])} befriend={int(step['befriend_count'])} betray={int(step['betray_count'])}"
        )
        for out in outcomes:
            agents = out["agents"]
            actions = out["actions"]
            rewards = out["rewards"]
            lines.append(
                f"   A{int(agents[0])}->{int(actions[0])} / A{int(agents[1])}->{int(actions[1])} "
                f"reward=({float(rewards[0]):+5.1f},{float(rewards[1]):+5.1f}) outcome={out['outcome']}"
            )
    return "\n".join(lines)


def print_recent_interactions(history: Sequence[Dict[str, object]], tail: int = 5) -> None:
    print(f"\n{format_recent_interactions(history, tail=tail)}")


def pause_menu(
    trainer: MiniSocietyTrainer,
    tracker: EpisodeTracker,
    current_episode: int,
    total_episodes: int,
    default_checkpoint_path: str,
    global_steps: int = 0,
    global_scores: Optional[Sequence[float]] = None,
) -> str:
    episode_label = (
        "continuous"
        if trainer.env.episode_length <= 0
        else f"{current_episode}/{total_episodes}"
    )
    print(
        f"\nPaused at episode {episode_label}, "
        f"step {trainer.env.timestep}/{horizon_text(trainer.env.episode_length)}."
    )
    print(f"Global steps: {global_steps}")
    print("Commands: resume, status, history [K], save [PATH], restart, quit, help")
    while True:
        raw = input("(paused)> ").strip()
        cmd = raw.split()
        if len(cmd) == 0:
            return "resume"
        head = cmd[0].lower()
        if head in {"resume", "r"}:
            return "resume"
        if head == "status":
            print_agent_state_snapshot(trainer)
            continue
        if head == "history":
            tail = 5
            if len(cmd) > 1:
                try:
                    tail = max(1, int(cmd[1]))
                except ValueError:
                    print("history expects an integer window, e.g. history 10")
                    continue
            print_recent_interactions(tracker.step_history, tail=tail)
            continue
        if head == "save":
            path = default_checkpoint_path if len(cmd) == 1 else cmd[1]
            save_checkpoint(
                path=path,
                trainer=trainer,
                current_episode=current_episode,
                total_episodes=total_episodes,
                tracker=tracker,
                global_steps=global_steps,
                global_scores=global_scores,
            )
            print(f"Checkpoint saved: {path}")
            continue
        if head == "restart":
            return "restart"
        if head == "quit":
            return "quit"
        if head == "help":
            print("resume: continue simulation")
            print("status: print current trust/confidence/incoming proposals for all agents")
            print("history [K]: show the last K interaction steps from current episode")
            print("save [PATH]: save full simulation checkpoint")
            print("restart: restart training from scratch (new env + new agent params)")
            print("quit: exit run loop")
            continue
        print(f"Unknown command: {head}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.n_agents < 2:
        raise ValueError("--n-agents must be at least 2.")
    if args.human_player and (args.human_agent_id < 0 or args.human_agent_id >= args.n_agents):
        raise ValueError(f"--human-agent-id must be in [0, {args.n_agents - 1}] when --human-player is enabled.")
    if args.human_player and not args.gui:
        raise ValueError("--human-player currently requires --gui for interactive decisions.")
    if args.episode_length < 0:
        raise ValueError("--episode-length must be non-negative (0 means unbounded).")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")
    if args.ppo_epochs <= 0:
        raise ValueError("--ppo-epochs must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive.")
    if not (0.0 <= args.gae_lambda <= 1.0):
        raise ValueError("--gae-lambda must be in [0, 1].")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative.")
    if args.coop_window <= 0:
        raise ValueError("--coop-window must be positive.")
    if args.reward_window <= 0:
        raise ValueError("--reward-window must be positive.")
    if args.viz_pause < 0:
        raise ValueError("--viz-pause must be non-negative.")
    if args.pause_poll_interval < 0:
        raise ValueError("--pause-poll-interval must be non-negative.")
    if args.gui_step_ms <= 0:
        raise ValueError("--gui-step-ms must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.clip_epsilon <= 0:
        raise ValueError("--clip-epsilon must be positive.")
    if args.value_coef < 0:
        raise ValueError("--value-coef must be non-negative.")
    if args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be non-negative.")
    if args.gossip_noise < 0:
        raise ValueError("--gossip-noise must be non-negative.")
    if not (0.0 <= args.gossip_strength <= 1.0):
        raise ValueError("--gossip-strength must be in [0, 1].")
    if args.confidence_decay < 0:
        raise ValueError("--confidence-decay must be non-negative.")
    if not (0.0 <= args.initial_confidence <= 1.0):
        raise ValueError("--initial-confidence must be in [0, 1].")
    if not (0.0 <= args.conf_gain_interaction <= 1.0):
        raise ValueError("--conf-gain-interaction must be in [0, 1].")
    if not (0.0 <= args.gamma <= 1.0):
        raise ValueError("--gamma must be in [0, 1].")
    if args.load_mode not in {"full", "weights_only"}:
        raise ValueError("--load-mode must be one of: full, weights_only.")


def build_trainer_from_args(args: argparse.Namespace, payoff: PayoffConfig) -> MiniSocietyTrainer:
    return MiniSocietyTrainer(
        n_agents=args.n_agents,
        episode_length=args.episode_length,
        confidence_decay=args.confidence_decay,
        gossip_noise=args.gossip_noise,
        gossip_strength=args.gossip_strength,
        initial_confidence=args.initial_confidence,
        reputation_delta_befriend=args.rep_delta_befriend,
        reputation_delta_betray=args.rep_delta_betray,
        confidence_gain_interaction=args.conf_gain_interaction,
        payoff=payoff,
        gamma=args.gamma,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        hidden_size=args.hidden_size,
        max_grad_norm=args.max_grad_norm,
        gae_lambda=args.gae_lambda,
        rollout_steps=args.rollout_steps,
        normalize_rewards=args.normalize_rewards,
        normalize_advantages=args.normalize_advantages,
        human_agent_id=(args.human_agent_id if args.human_player else None),
        device=args.device,
    )


def build_payoff_from_args(args: argparse.Namespace) -> PayoffConfig:
    return PayoffConfig(
        mutual_befriend=(float(args.payoff_mutual_befriend[0]), float(args.payoff_mutual_befriend[1])),
        betray_befriend=(float(args.payoff_betray_befriend[0]), float(args.payoff_betray_befriend[1])),
        befriend_betray=(float(args.payoff_befriend_betray[0]), float(args.payoff_befriend_betray[1])),
        mutual_betray=(float(args.payoff_mutual_betray[0]), float(args.payoff_mutual_betray[1])),
        isolation=float(args.isolation_penalty),
    )


def load_into_trainer(
    args: argparse.Namespace,
    trainer: MiniSocietyTrainer,
    checkpoint_path: str,
) -> Tuple[MiniSocietyTrainer, int, Optional[EpisodeTracker], int, Optional[List[float]]]:
    current_episode = 1
    global_steps = 0
    global_scores: Optional[List[float]] = None
    tracker: Optional[EpisodeTracker] = None
    loaded_checkpoint = load_checkpoint(checkpoint_path, device=args.device)
    trainer_state = loaded_checkpoint["trainer_state"]

    if args.load_mode == "full":
        loaded_config = dict(trainer_state["config"])  # type: ignore[index]
        loaded_config["device"] = args.device
        if args.human_player:
            loaded_config["human_agent_id"] = int(args.human_agent_id)
        trainer = MiniSocietyTrainer.from_config(loaded_config)
        trainer.load_state(trainer_state, load_optimizer=True, load_buffers=True, load_env=True)

        run_state = loaded_checkpoint.get("run_state", {})
        current_episode = int(run_state.get("current_episode", 1))
        global_steps = int(run_state.get("global_steps", trainer.env.timestep))
        loaded_scores = run_state.get("global_scores")
        if loaded_scores is not None:
            global_scores = [float(v) for v in loaded_scores]
        saved_total_episodes = int(run_state.get("total_episodes", args.episodes))
        if args.episodes < current_episode:
            args.episodes = max(saved_total_episodes, current_episode)
        tracker_payload = run_state.get("tracker")
        if tracker_payload is not None:
            tracker = deserialize_episode_tracker(tracker_payload)
            if global_scores is None:
                global_scores = [float(v) for v in tracker.episode_returns]

        rng_state = loaded_checkpoint.get("rng_state")
        if rng_state is not None:
            _restore_rng_state(rng_state)
    else:
        trainer.load_state(trainer_state, load_optimizer=False, load_buffers=False, load_env=False)
        if args.transfer_trust_priors:
            env_state = trainer_state["env_state"]  # type: ignore[index]
            trust_stack = env_state["trust_stack"]  # type: ignore[index]
            reputation_prior = trust_stack[:, :, 0]
            confidence_prior = trust_stack[:, :, 1]
            trainer.env.set_initial_priors(reputation_prior, confidence_prior)
        trainer.env.reset()
        for agent in trainer.agents:
            agent.buffer.clear()
        global_steps = 0

    if global_scores is not None and len(global_scores) != trainer.n_agents:
        global_scores = global_scores[: trainer.n_agents]
        if len(global_scores) < trainer.n_agents:
            global_scores.extend([0.0 for _ in range(trainer.n_agents - len(global_scores))])

    return trainer, current_episode, tracker, global_steps, global_scores


def format_episode_summary(episode: int, result: Dict[str, object]) -> str:
    returns = ", ".join(f"{r:6.1f}" for r in result["episode_returns"])  # type: ignore[index]
    updates = int(result.get("updates", 0))
    steps = int(result.get("steps", 0))
    return (
        f"ep={episode:04d} "
        f"mean_return={result['mean_return']:7.2f} "
        f"steps={steps:4d} "
        f"pairs={result['pair_count']:3d} "
        f"loss={result['mean_loss']:8.4f} "
        f"entropy={result['mean_entropy']:6.3f} "
        f"updates={updates:3d} "
        f"returns=[{returns}]"
    )


def format_run_label(segment_index: int, continuous_mode: bool) -> str:
    if continuous_mode:
        return f"run=cont segment={segment_index:04d}"
    return f"ep={segment_index:04d}"


class MiniSocietyGUIApp:
    def __init__(self, args: argparse.Namespace, payoff: PayoffConfig) -> None:
        self.args = args
        self.payoff = payoff
        self.continuous_mode = bool(args.continuous or args.episode_length == 0)
        self.total_episodes = args.episodes
        self.current_episode = 1
        self.global_steps = 0
        self.global_scores: List[float] = []
        self.tracker: Optional[EpisodeTracker] = None
        self.last_history: List[Dict[str, object]] = []
        self.running = False
        self.paused = False
        self.after_id = None
        self.pending_human_step: Optional[PreparedHumanStep] = None
        self.hyperparams_window = None
        self.hyperparam_vars: Dict[str, object] = {}
        self.hyperparam_meta: Dict[str, Dict[str, object]] = {}

        self.trainer = build_trainer_from_args(args, payoff)
        self.human_mode = self.trainer.human_agent_id is not None
        if args.load_checkpoint.strip():
            (
                self.trainer,
                self.current_episode,
                self.tracker,
                self.global_steps,
                loaded_scores,
            ) = load_into_trainer(
                args=args,
                trainer=self.trainer,
                checkpoint_path=args.load_checkpoint.strip(),
            )
            self.total_episodes = args.episodes
            self.human_mode = self.trainer.human_agent_id is not None
            if loaded_scores is not None:
                self.global_scores = loaded_scores
        if len(self.global_scores) != self.trainer.n_agents:
            self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]

        try:
            import tkinter as tk
            from tkinter import messagebox, scrolledtext, ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise RuntimeError(
                "GUI mode requires tkinter and matplotlib Tk backend. "
                "Install GUI dependencies and use a Python build with tkinter."
            ) from exc

        self.tk = tk
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.ttk = ttk
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        self.Figure = Figure

        self.root = self.tk.Tk()
        self.root.title("MiniSociety MARL Control Panel")
        self.root.geometry("1500x900")
        self.root.minsize(1280, 900)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.status_var = self.tk.StringVar()
        self._build_ui()
        self.root.bind("<Configure>", self._on_root_resize)
        self._sync_human_controls_ui()
        self._update_status()
        self._update_leaderboard()
        self._render_if_available()

    def _build_ui(self) -> None:
        controls = self.tk.Frame(self.root)
        controls.pack(fill=self.tk.X, padx=8, pady=6)
        controls_top = self.tk.Frame(controls)
        controls_top.pack(fill=self.tk.X)
        controls_bottom = self.tk.Frame(controls)
        controls_bottom.pack(fill=self.tk.X, pady=(4, 0))

        self.start_button = self.tk.Button(controls_top, text="Start", width=11, command=self.on_start)
        self.start_button.pack(side=self.tk.LEFT, padx=2)
        self.pause_button = self.tk.Button(
            controls_top, text="Pause", width=11, state=self.tk.DISABLED, command=self.on_pause_resume
        )
        self.pause_button.pack(side=self.tk.LEFT, padx=2)
        self.status_button = self.tk.Button(controls_top, text="Status", width=11, command=self.on_status)
        self.status_button.pack(side=self.tk.LEFT, padx=2)

        self.history_k_var = self.tk.StringVar(value="5")
        self.tk.Label(controls_top, text="History K:").pack(side=self.tk.LEFT, padx=(12, 2))
        self.history_k_entry = self.tk.Entry(controls_top, textvariable=self.history_k_var, width=5)
        self.history_k_entry.pack(side=self.tk.LEFT, padx=2)
        self.history_button = self.tk.Button(controls_top, text="History", width=11, command=self.on_history)
        self.history_button.pack(side=self.tk.LEFT, padx=2)

        self.save_path_var = self.tk.StringVar(value=self.args.checkpoint_path)
        self.tk.Label(controls_bottom, text="Save Path:").pack(side=self.tk.LEFT, padx=(0, 2))
        self.save_path_entry = self.tk.Entry(controls_bottom, textvariable=self.save_path_var)
        self.save_path_entry.pack(side=self.tk.LEFT, padx=2, fill=self.tk.X, expand=True)
        self.save_button = self.tk.Button(controls_bottom, text="Save", width=11, command=self.on_save)
        self.save_button.pack(side=self.tk.LEFT, padx=2)

        self.restart_button = self.tk.Button(controls_bottom, text="Restart", width=11, command=self.on_restart)
        self.restart_button.pack(side=self.tk.LEFT, padx=2)
        self.quit_button = self.tk.Button(controls_bottom, text="Quit", width=11, command=self.on_quit)
        self.quit_button.pack(side=self.tk.LEFT, padx=2)
        self.help_button = self.tk.Button(controls_bottom, text="Help", width=11, command=self.on_help)
        self.help_button.pack(side=self.tk.LEFT, padx=2)
        self.hyperparams_button = self.tk.Button(
            controls_bottom,
            text="Hyperparams",
            width=12,
            command=self.on_open_hyperparams,
        )
        self.hyperparams_button.pack(side=self.tk.LEFT, padx=2)

        status_frame = self.tk.Frame(self.root)
        status_frame.pack(fill=self.tk.X, padx=8, pady=(0, 4))
        self.status_label = self.tk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            justify=self.tk.LEFT,
        )
        self.status_label.pack(fill=self.tk.X)
        self.leaderboard_var = self.tk.StringVar(value="")
        self.leaderboard_label = self.tk.Label(
            status_frame,
            textvariable=self.leaderboard_var,
            anchor="w",
            justify=self.tk.LEFT,
            font=("Consolas", 10),
        )
        self.leaderboard_label.pack(fill=self.tk.X)

        self.human_frame = self.tk.Frame(self.root)
        self.human_frame.pack(fill=self.tk.X, padx=8, pady=(0, 4))
        self.human_label_var = self.tk.StringVar(value="Human Player: disabled")
        self.tk.Label(self.human_frame, textvariable=self.human_label_var).pack(side=self.tk.LEFT, padx=(0, 8))

        propose_values = self._human_propose_values()
        self.human_propose_var = self.tk.StringVar(value=propose_values[-1] if propose_values else "None")
        self.tk.Label(self.human_frame, text="Propose:").pack(side=self.tk.LEFT, padx=(4, 2))
        self.human_propose_combo = self.ttk.Combobox(
            self.human_frame,
            state="readonly",
            width=14,
            textvariable=self.human_propose_var,
            values=propose_values if len(propose_values) > 0 else ["None"],
        )
        self.human_propose_combo.pack(side=self.tk.LEFT, padx=2)

        self.human_response_var = self.tk.StringVar(value="Decline")
        self.tk.Label(self.human_frame, text="Respond:").pack(side=self.tk.LEFT, padx=(8, 2))
        self.human_response_combo = self.ttk.Combobox(
            self.human_frame,
            state="readonly",
            width=14,
            textvariable=self.human_response_var,
            values=["Decline"],
        )
        self.human_response_combo.pack(side=self.tk.LEFT, padx=2)

        self.human_play_var = self.tk.StringVar(value="Befriend")
        self.tk.Label(self.human_frame, text="Play:").pack(side=self.tk.LEFT, padx=(8, 2))
        self.human_play_combo = self.ttk.Combobox(
            self.human_frame,
            state="readonly",
            width=12,
            textvariable=self.human_play_var,
            values=["Betray", "Befriend"],
        )
        self.human_play_combo.pack(side=self.tk.LEFT, padx=2)

        self.prepare_human_button = self.tk.Button(
            self.human_frame,
            text="Prepare Step",
            width=14,
            state=self.tk.DISABLED,
            command=self.on_prepare_human_step,
        )
        self.prepare_human_button.pack(side=self.tk.LEFT, padx=(10, 2))
        self.commit_human_button = self.tk.Button(
            self.human_frame,
            text="Commit Step",
            width=12,
            state=self.tk.DISABLED,
            command=self.on_commit_human_step,
        )
        self.commit_human_button.pack(side=self.tk.LEFT, padx=2)

        viz_frame = self.tk.Frame(self.root)
        viz_frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=4)
        self.figure = self.Figure(figsize=(14.5, 6.8), dpi=100)
        axes = self.figure.subplots(1, 3, gridspec_kw={"width_ratios": [1.05, 1.15, 1.2]})
        self.ax_graph, self.ax_heat, self.ax_metrics = axes
        self.ax_metrics_right = self.ax_metrics.twinx()
        self.canvas = self.FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=self.tk.BOTH, expand=True)
        self.toolbar = self.NavigationToolbar2Tk(self.canvas, viz_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=self.tk.X)

        log_frame = self.tk.Frame(self.root)
        log_frame.pack(fill=self.tk.BOTH, expand=False, padx=8, pady=4)
        self.log_box = self.scrolledtext.ScrolledText(log_frame, height=12, wrap=self.tk.WORD)
        self.log_box.pack(fill=self.tk.BOTH, expand=True)

        self._bind_visualizer()

    def _on_root_resize(self, event: object) -> None:
        if getattr(event, "widget", None) is not self.root:
            return
        width = max(720, int(getattr(event, "width", 0)) - 24)
        self.status_label.configure(wraplength=width)
        self.leaderboard_label.configure(wraplength=width)

    def _bind_visualizer(self) -> None:
        if hasattr(self, "visualizer") and self.visualizer is not None and self.visualizer.heat_colorbar is not None:
            try:
                self.visualizer.heat_colorbar.remove()
            except Exception:
                pass
        self.visualizer = MiniSocietyVisualizer(
            n_agents=self.trainer.n_agents,
            human_agent_id=self.trainer.human_agent_id,
            cooperation_window=self.args.coop_window,
            reward_window=self.args.reward_window,
            pause_seconds=0.0,
        )
        self.visualizer.fig = self.figure
        self.visualizer.ax_graph = self.ax_graph
        self.visualizer.ax_heat = self.ax_heat
        self.visualizer.ax_metrics = self.ax_metrics
        self.visualizer.ax_metrics_right = self.ax_metrics_right
        self.visualizer.heat_colorbar = None
        self.visualizer._live_interactive_backend = False

    def _sync_human_controls_ui(self) -> None:
        if self.human_mode and self.trainer.human_agent_id is not None:
            self.human_label_var.set(f"Human Player: A{self.trainer.human_agent_id}")
            propose_values = self._human_propose_values()
            self.human_propose_combo.configure(values=propose_values if len(propose_values) > 0 else ["None"])
            if self.human_propose_var.get() not in propose_values:
                self.human_propose_var.set(propose_values[-1] if len(propose_values) > 0 else "None")
            self.human_response_combo.configure(values=["Decline"])
            self.human_response_var.set("Decline")
            self.human_play_var.set("Befriend")
            self.human_propose_combo.configure(state="readonly")
            self.human_response_combo.configure(state="readonly")
            self.human_play_combo.configure(state="readonly")
        else:
            self.human_label_var.set("Human Player: disabled")
            self.human_propose_combo.configure(values=["None"])
            self.human_propose_var.set("None")
            self.human_response_combo.configure(values=["Decline"])
            self.human_response_var.set("Decline")
            self.human_play_var.set("Befriend")
            self.human_propose_combo.configure(state="disabled")
            self.human_response_combo.configure(state="disabled")
            self.human_play_combo.configure(state="disabled")
        self.prepare_human_button.config(state=self.tk.DISABLED)
        self.commit_human_button.config(state=self.tk.DISABLED)

    def _clear_visualization(self) -> None:
        self.ax_graph.clear()
        self.ax_heat.clear()
        self.ax_metrics.clear()
        self.ax_metrics_right.clear()
        self.figure.suptitle("")
        self.canvas.draw_idle()

    def _reset_simulation_from_args(self, reason: str) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        set_seed(self.args.seed)
        self.payoff = build_payoff_from_args(self.args)
        self.trainer = build_trainer_from_args(self.args, self.payoff)
        self.human_mode = self.trainer.human_agent_id is not None
        self.continuous_mode = bool(self.args.continuous or self.args.episode_length == 0)
        self.total_episodes = self.args.episodes
        self.current_episode = 1
        self.global_steps = 0
        self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]
        self.tracker = None
        self.last_history = []
        self.pending_human_step = None
        self.running = False
        self.paused = False
        self.start_button.config(state=self.tk.NORMAL)
        self.pause_button.config(state=self.tk.DISABLED, text="Pause")
        self.save_path_var.set(self.args.checkpoint_path)
        self._bind_visualizer()
        self._sync_human_controls_ui()
        self._clear_visualization()
        self._update_status()
        self._update_leaderboard()
        self.log(reason)

    def _schedule_next(self, delay_ms: Optional[int] = None) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        interval = max(1, int(self.args.gui_step_ms))
        self.after_id = self.root.after(interval if delay_ms is None else delay_ms, self._step_loop)

    def on_open_hyperparams(self) -> None:
        if self.hyperparams_window is not None and self.hyperparams_window.winfo_exists():
            self.hyperparams_window.lift()
            self.hyperparams_window.focus_force()
            return

        window = self.tk.Toplevel(self.root)
        window.title("Hyperparameter Editor")
        window.geometry("760x760")
        window.transient(self.root)
        self.hyperparams_window = window
        self.hyperparam_vars = {}
        self.hyperparam_meta = {}

        def _close_window() -> None:
            self.hyperparams_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", _close_window)

        container = self.tk.Frame(window)
        container.pack(fill=self.tk.BOTH, expand=True)
        canvas = self.tk.Canvas(container, borderwidth=0)
        scrollbar = self.tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        form = self.tk.Frame(canvas)
        form.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self.tk.Label(
            form,
            text="Edit values, then Apply + Restart. Lists use comma-separated values.",
            anchor="w",
            justify=self.tk.LEFT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 10))

        args_dict = vars(self.args)
        for idx, key in enumerate(args_dict.keys(), start=1):
            value = args_dict[key]
            self.tk.Label(form, text=key, anchor="w").grid(row=idx, column=0, sticky="w", padx=8, pady=3)
            if isinstance(value, bool):
                var = self.tk.BooleanVar(value=bool(value))
                widget = self.tk.Checkbutton(form, variable=var)
                widget.grid(row=idx, column=1, sticky="w", padx=8, pady=3)
                self.hyperparam_vars[key] = var
                self.hyperparam_meta[key] = {"kind": "bool"}
            else:
                if isinstance(value, list):
                    text_value = ", ".join(str(v) for v in value)
                    kind = "list"
                    list_len = len(value)
                else:
                    text_value = str(value)
                    kind = "scalar"
                    list_len = 0
                var = self.tk.StringVar(value=text_value)
                entry = self.tk.Entry(form, textvariable=var, width=42)
                entry.grid(row=idx, column=1, sticky="we", padx=8, pady=3)
                self.hyperparam_vars[key] = var
                self.hyperparam_meta[key] = {
                    "kind": kind,
                    "py_type": type(value),
                    "list_len": list_len,
                }
                if key == "gui":
                    entry.configure(state="disabled")
                    self.tk.Label(form, text="(fixed in GUI mode)", anchor="w").grid(
                        row=idx, column=2, sticky="w", padx=4
                    )

        form.grid_columnconfigure(1, weight=1)

        buttons = self.tk.Frame(window)
        buttons.pack(fill=self.tk.X, padx=8, pady=8)
        self.tk.Button(buttons, text="Apply + Restart", width=16, command=self.on_apply_hyperparams).pack(
            side=self.tk.LEFT, padx=4
        )
        self.tk.Button(buttons, text="Reload Current", width=14, command=self.on_reload_hyperparams).pack(
            side=self.tk.LEFT, padx=4
        )
        self.tk.Button(buttons, text="Close", width=10, command=_close_window).pack(side=self.tk.RIGHT, padx=4)

    def on_reload_hyperparams(self) -> None:
        if self.hyperparams_window is not None and self.hyperparams_window.winfo_exists():
            self.hyperparams_window.destroy()
        self.hyperparams_window = None
        self.on_open_hyperparams()

    def on_apply_hyperparams(self) -> None:
        if self.hyperparams_window is None or not self.hyperparams_window.winfo_exists():
            return

        new_args = argparse.Namespace(**dict(vars(self.args)))
        try:
            for key, meta in self.hyperparam_meta.items():
                kind = meta["kind"]
                raw_var = self.hyperparam_vars[key]
                if kind == "bool":
                    setattr(new_args, key, bool(raw_var.get()))  # type: ignore[union-attr]
                    continue

                text = str(raw_var.get()).strip()  # type: ignore[union-attr]
                if kind == "list":
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    expected_len = int(meta["list_len"])
                    if len(parts) != expected_len:
                        raise ValueError(f"{key} expects {expected_len} comma-separated values.")
                    setattr(new_args, key, [float(p) for p in parts])
                    continue

                py_type = meta["py_type"]
                if py_type is int:
                    setattr(new_args, key, int(text))
                elif py_type is float:
                    setattr(new_args, key, float(text))
                else:
                    setattr(new_args, key, text)
        except ValueError as exc:
            self.messagebox.showerror("Invalid Hyperparameter", str(exc))
            return

        new_args.gui = True
        if bool(new_args.continuous):
            new_args.episode_length = 0

        try:
            validate_args(new_args)
            build_payoff_from_args(new_args)
        except Exception as exc:
            self.messagebox.showerror("Validation Error", str(exc))
            return

        if self.running or self.tracker is not None:
            confirm = self.messagebox.askyesno(
                "Apply Hyperparameters",
                "Apply new hyperparameters and restart simulation now?",
            )
            if not confirm:
                return

        self.args = new_args
        try:
            self._reset_simulation_from_args("Applied hyperparameters from UI and restarted simulation.")
        except Exception as exc:
            self.messagebox.showerror("Apply Failed", str(exc))
            return

        if self.hyperparams_window is not None and self.hyperparams_window.winfo_exists():
            self.hyperparams_window.destroy()
            self.hyperparams_window = None

    def _human_propose_values(self) -> List[str]:
        if not self.human_mode:
            return []
        assert self.trainer.human_agent_id is not None
        values = [f"A{i}" for i in range(self.trainer.n_agents) if i != self.trainer.human_agent_id]
        values.append("None")
        return values

    def _parse_human_agent_choice(self, raw: str, decline_token: str) -> int:
        value = raw.strip()
        if value == decline_token:
            return self.trainer.n_agents
        if value.startswith("A"):
            return int(value[1:])
        raise ValueError(f"Invalid agent choice: {value}")

    @staticmethod
    def _parse_human_play_choice(raw: str) -> int:
        value = raw.strip().lower()
        if value == "betray":
            return 0
        if value == "befriend":
            return 1
        raise ValueError(f"Invalid play choice: {raw}")

    def _accumulate_global_scores(self, rewards: Sequence[float]) -> None:
        if len(self.global_scores) != self.trainer.n_agents:
            self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]
        for i, value in enumerate(rewards):
            self.global_scores[i] += float(value)

    def _update_leaderboard(self) -> None:
        self.leaderboard_var.set(
            format_gui_leaderboard(
                self.global_scores,
                human_mode=self.human_mode,
                human_agent_id=self.trainer.human_agent_id,
            )
        )

    def _update_status(self) -> None:
        phase = "running" if self.running and not self.paused else "paused" if self.paused else "idle"
        step_now = self.trainer.env.timestep
        episode_text = "continuous" if self.continuous_mode else f"{self.current_episode}/{self.total_episodes}"
        mode_text = "human" if self.human_mode else "ai-only"
        pending_text = "yes" if self.pending_human_step is not None else "no"
        self.status_var.set(
            format_gui_status(
                phase=phase,
                mode_text=mode_text,
                episode_text=episode_text,
                step_now=step_now,
                episode_length=self.trainer.env.episode_length,
                global_steps=self.global_steps,
                pending_text=pending_text,
            )
        )

    def _get_history_source(self) -> List[Dict[str, object]]:
        if self.tracker is not None and len(self.tracker.step_history) > 0:
            return self.tracker.step_history
        return self.last_history

    def _render_if_available(self) -> None:
        history = self._get_history_source()
        if len(history) == 0:
            return
        step_idx = len(history)
        total_steps = self.trainer.env.episode_length if self.trainer.env.episode_length > 0 else step_idx
        step_data = history[-1]
        self.visualizer._draw_social_graph(step_data, step_idx=step_idx, total_steps=total_steps)
        self.visualizer._draw_trust_heatmap(step_data)
        self.visualizer._draw_metrics(history)
        self.visualizer._apply_layout("Mini-Society Dashboard (Live)")
        self.canvas.draw_idle()

    def log(self, message: str) -> None:
        self.log_box.insert(self.tk.END, message + "\n")
        self.log_box.see(self.tk.END)

    def _finish_current_episode(self) -> None:
        if self.tracker is None:
            return
        result = self.trainer.finish_episode(self.tracker, terminal=bool(self.tracker.done))
        if len(self.tracker.step_history) > 0:
            self.last_history = list(self.tracker.step_history)
        summary = format_episode_summary(self.current_episode, result)
        summary = summary.replace(
            f"ep={self.current_episode:04d}",
            format_run_label(self.current_episode, self.continuous_mode),
        )
        self.log(summary)
        self.tracker = None
        self.pending_human_step = None
        if self.human_mode:
            self.commit_human_button.config(state=self.tk.DISABLED)
            self.human_response_combo.configure(values=["Decline"])
            self.human_response_var.set("Decline")
        if not self.continuous_mode:
            self.current_episode += 1

    def _finalize_completed_run(self) -> None:
        self.running = False
        self.paused = False
        self.pending_human_step = None
        self.pause_button.config(state=self.tk.DISABLED, text="Pause")
        self.start_button.config(state=self.tk.NORMAL)
        if self.human_mode:
            self.prepare_human_button.config(state=self.tk.DISABLED)
            self.commit_human_button.config(state=self.tk.DISABLED)
        self._update_status()
        self._update_leaderboard()
        self.log("Run completed.")

        if self.args.save_checkpoint.strip():
            save_checkpoint(
                path=self.args.save_checkpoint.strip(),
                trainer=self.trainer,
                current_episode=self.current_episode,
                total_episodes=self.total_episodes,
                tracker=self.tracker,
                global_steps=self.global_steps,
                global_scores=self.global_scores,
            )
            self.log(f"Saved checkpoint: {self.args.save_checkpoint.strip()}")

        if self.args.viz_output.strip() and len(self.last_history) > 0:
            self.figure.savefig(self.args.viz_output.strip(), dpi=170, bbox_inches="tight")
            self.log(f"Saved visualization snapshot: {self.args.viz_output.strip()}")

    def _step_loop(self) -> None:
        self.after_id = None
        if not self.running or self.paused:
            return
        if self.human_mode:
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return
            if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
                self.log(f"Reached max global steps: {self.args.max_steps}")
                self._finalize_completed_run()
                return
            if self.tracker is None:
                self.tracker = self.trainer.start_episode(collect_history=True)
            self.prepare_human_button.config(state=self.tk.NORMAL)
            self.commit_human_button.config(
                state=self.tk.NORMAL if self.pending_human_step is not None else self.tk.DISABLED
            )
            self._update_status()
            return
        if not self.continuous_mode and self.current_episode > self.total_episodes:
            self._finalize_completed_run()
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)

        info = self.trainer.step_episode(self.tracker, collect_history=True)
        self._accumulate_global_scores(info["agent_rewards"])  # type: ignore[index]
        self.global_steps += 1
        self._update_leaderboard()
        self._render_if_available()
        self._update_status()

        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            if self.tracker is not None:
                # External step cap truncates run; treat as non-terminal for value bootstrap.
                self.tracker.done = False
                self._finish_current_episode()
            self.log(f"Reached max global steps: {self.args.max_steps}")
            self._finalize_completed_run()
            return

        if self.tracker.done:
            self._finish_current_episode()
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return

        if self.running and not self.paused:
            self._schedule_next()

    def on_start(self) -> None:
        if not self.continuous_mode and self.current_episode > self.total_episodes:
            self.log("All episodes are already complete. Click Restart to begin again.")
            return
        if self.running and not self.paused:
            return
        self.running = True
        self.paused = False
        self.start_button.config(state=self.tk.DISABLED)
        self.pause_button.config(state=self.tk.NORMAL, text="Pause")
        self._update_status()
        if self.human_mode:
            self.prepare_human_button.config(state=self.tk.NORMAL)
            self.commit_human_button.config(
                state=self.tk.NORMAL if self.pending_human_step is not None else self.tk.DISABLED
            )
            self._schedule_next(delay_ms=1)
            return
        self._schedule_next(delay_ms=1)

    def on_pause_resume(self) -> None:
        if not self.running:
            self.on_start()
            return
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            if self.human_mode:
                self.prepare_human_button.config(state=self.tk.DISABLED)
                self.commit_human_button.config(state=self.tk.DISABLED)
        else:
            self.pause_button.config(text="Pause")
            if self.human_mode:
                self.prepare_human_button.config(state=self.tk.NORMAL)
                self.commit_human_button.config(
                    state=self.tk.NORMAL if self.pending_human_step is not None else self.tk.DISABLED
                )
            self._schedule_next(delay_ms=1)
        self._update_status()

    def on_status(self) -> None:
        self.log(format_agent_state_snapshot(self.trainer))
        self.log(self.leaderboard_var.get())

    def on_history(self) -> None:
        try:
            tail = max(1, int(self.history_k_var.get().strip()))
        except ValueError:
            self.log("History K must be an integer.")
            return
        self.log(format_recent_interactions(self._get_history_source(), tail=tail))

    def on_prepare_human_step(self) -> None:
        if not self.human_mode:
            self.log("Human controls are unavailable in AI-only mode.")
            return
        if not self.running or self.paused:
            self.log("Start or resume the simulation before preparing a human step.")
            return
        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            self.log(f"Global step cap reached: {self.args.max_steps}")
            self._finalize_completed_run()
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)
        if self.tracker.done:
            self._finish_current_episode()
            self._update_status()
            return

        try:
            human_propose = self._parse_human_agent_choice(self.human_propose_var.get(), decline_token="None")
        except ValueError as exc:
            self.log(str(exc))
            return

        try:
            prepared = self.trainer.prepare_human_step(human_propose_action=human_propose)
        except Exception as exc:
            self.log(f"Failed to prepare human step: {exc}")
            return

        self.pending_human_step = prepared
        assert self.trainer.human_agent_id is not None
        human_id = self.trainer.human_agent_id
        mask = prepared.respond_masks[human_id].detach().cpu().tolist()
        response_values = [f"A{i}" for i in range(self.trainer.n_agents) if bool(mask[i])]
        response_values.append("Decline")
        self.human_response_combo.configure(values=response_values)
        self.human_response_var.set("Decline")
        self.commit_human_button.config(state=self.tk.NORMAL)

        incoming = [
            i
            for i, v in enumerate(self.trainer.env.states[human_id].incoming_proposals.detach().cpu().tolist())
            if int(v) == 1
        ]
        incoming_text = "none" if len(incoming) == 0 else ", ".join(f"A{i}" for i in incoming)
        self.log(
            f"Prepared step t={prepared.timestep}: human propose={self.human_propose_var.get()} "
            f"| incoming proposals -> {incoming_text}"
        )
        self._update_status()

    def on_commit_human_step(self) -> None:
        if not self.human_mode:
            self.log("Human controls are unavailable in AI-only mode.")
            return
        if not self.running or self.paused:
            self.log("Start or resume the simulation before committing a human step.")
            return
        if self.pending_human_step is None:
            self.log("No prepared human step. Click 'Prepare Step' first.")
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)

        try:
            human_response = self._parse_human_agent_choice(self.human_response_var.get(), decline_token="Decline")
            human_play = self._parse_human_play_choice(self.human_play_var.get())
        except ValueError as exc:
            self.log(str(exc))
            return

        prepared = self.pending_human_step
        try:
            info = self.trainer.execute_prepared_human_step(
                tracker=self.tracker,
                prepared=prepared,
                human_response_action=human_response,
                human_play_action=human_play,
                collect_history=True,
            )
        except Exception as exc:
            self.log(f"Failed to commit human step: {exc}")
            return

        self.pending_human_step = None
        self.commit_human_button.config(state=self.tk.DISABLED)
        self._accumulate_global_scores(info["agent_rewards"])  # type: ignore[index]
        self.global_steps += 1
        self._update_leaderboard()
        self._render_if_available()
        self._update_status()

        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            if self.tracker is not None:
                self.tracker.done = False
                self._finish_current_episode()
            self.log(f"Reached max global steps: {self.args.max_steps}")
            self._finalize_completed_run()
            return

        if self.tracker is not None and self.tracker.done:
            self._finish_current_episode()
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return

    def on_save(self) -> None:
        path = self.save_path_var.get().strip() or self.args.checkpoint_path
        if not path:
            self.log("No save path provided.")
            return
        save_checkpoint(
            path=path,
            trainer=self.trainer,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            tracker=self.tracker,
            global_steps=self.global_steps,
            global_scores=self.global_scores,
        )
        self.log(f"Checkpoint saved: {path}")

    def on_restart(self) -> None:
        if not self.messagebox.askyesno("Restart", "Restart from scratch? Current unsaved progress will be lost."):
            return
        self._reset_simulation_from_args("Simulation restarted from scratch.")

    def on_help(self) -> None:
        self.log("start: begin simulation")
        self.log("pause/resume: pause after current step or continue")
        self.log("continuous mode: launch with --continuous for a single uninterrupted run")
        self.log("leaderboard: cumulative scores across all elapsed steps")
        if self.human_mode:
            self.log("human mode: choose propose/respond/play each step")
            self.log("prepare step: samples AI proposals/responses and updates your valid response options")
            self.log("commit step: executes one full environment step with your selected response/play")
        self.log("status: print current trust/confidence/incoming proposals for all agents")
        self.log("history K: show latest K interaction steps")
        self.log("save: write full checkpoint to save path")
        self.log("hyperparams: open editor to view/change all args, then apply + restart")
        self.log("restart: reset env and agents from scratch")
        self.log("quit: close app")

    def on_quit(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


class MiniSocietyQtApp:
    def __init__(self, args: argparse.Namespace, payoff: PayoffConfig) -> None:
        self.args = args
        self.payoff = payoff
        self.continuous_mode = bool(args.continuous or args.episode_length == 0)
        self.total_episodes = args.episodes
        self.current_episode = 1
        self.global_steps = 0
        self.global_scores: List[float] = []
        self.tracker: Optional[EpisodeTracker] = None
        self.last_history: List[Dict[str, object]] = []
        self.running = False
        self.paused = False
        self.pending_human_step: Optional[PreparedHumanStep] = None
        self.hyperparams_dialog = None
        self.hyperparam_widgets: Dict[str, Dict[str, object]] = {}

        self.trainer = build_trainer_from_args(args, payoff)
        self.human_mode = self.trainer.human_agent_id is not None
        if args.load_checkpoint.strip():
            (
                self.trainer,
                self.current_episode,
                self.tracker,
                self.global_steps,
                loaded_scores,
            ) = load_into_trainer(
                args=args,
                trainer=self.trainer,
                checkpoint_path=args.load_checkpoint.strip(),
            )
            self.total_episodes = args.episodes
            self.human_mode = self.trainer.human_agent_id is not None
            if loaded_scores is not None:
                self.global_scores = loaded_scores
        if len(self.global_scores) != self.trainer.n_agents:
            self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]

        try:
            from PySide6 import QtCore, QtWidgets
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise RuntimeError(
                "PySide6 GUI requires PySide6 and matplotlib Qt backend. "
                "Install with: pip install PySide6 matplotlib"
            ) from exc

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.FigureCanvasQTAgg = FigureCanvasQTAgg
        self.NavigationToolbar2QT = NavigationToolbar2QT
        self.Figure = Figure

        app = QtWidgets.QApplication.instance()
        self.app = app if app is not None else QtWidgets.QApplication([])

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("MiniSociety MARL Control Panel (PySide6)")
        self.window.resize(1560, 980)
        self.window.setMinimumSize(1320, 920)
        self.window.closeEvent = self._on_close_event  # type: ignore[assignment]

        self._build_ui()
        self._bind_visualizer()
        self._sync_human_controls_ui()
        self._update_status()
        self._update_leaderboard()
        self._render_if_available()

    def _build_ui(self) -> None:
        QtWidgets = self.QtWidgets
        QtCore = self.QtCore

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls_top = QtWidgets.QHBoxLayout()
        root.addLayout(controls_top)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.on_start)
        controls_top.addWidget(self.start_button)

        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.on_pause_resume)
        controls_top.addWidget(self.pause_button)

        self.status_button = QtWidgets.QPushButton("Status")
        self.status_button.clicked.connect(self.on_status)
        controls_top.addWidget(self.status_button)

        controls_top.addSpacing(8)
        controls_top.addWidget(QtWidgets.QLabel("History K:"))
        self.history_k_spin = QtWidgets.QSpinBox()
        self.history_k_spin.setRange(1, 1_000_000)
        self.history_k_spin.setValue(5)
        self.history_k_spin.setFixedWidth(80)
        controls_top.addWidget(self.history_k_spin)
        self.history_button = QtWidgets.QPushButton("History")
        self.history_button.clicked.connect(self.on_history)
        controls_top.addWidget(self.history_button)
        controls_top.addStretch(1)

        controls_bottom = QtWidgets.QHBoxLayout()
        root.addLayout(controls_bottom)
        controls_bottom.addWidget(QtWidgets.QLabel("Save Path:"))
        self.save_path_edit = QtWidgets.QLineEdit(self.args.checkpoint_path)
        self.save_path_edit.setMinimumWidth(340)
        controls_bottom.addWidget(self.save_path_edit, stretch=1)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.on_save)
        controls_bottom.addWidget(self.save_button)

        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.clicked.connect(self.on_restart)
        controls_bottom.addWidget(self.restart_button)

        self.quit_button = QtWidgets.QPushButton("Quit")
        self.quit_button.clicked.connect(self.on_quit)
        controls_bottom.addWidget(self.quit_button)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.on_help)
        controls_bottom.addWidget(self.help_button)

        self.hyperparams_button = QtWidgets.QPushButton("Hyperparams")
        self.hyperparams_button.clicked.connect(self.on_open_hyperparams)
        controls_bottom.addWidget(self.hyperparams_button)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-family: Consolas, monospace;")
        root.addWidget(self.status_label)

        self.leaderboard_label = QtWidgets.QLabel("")
        self.leaderboard_label.setWordWrap(True)
        self.leaderboard_label.setStyleSheet("font-family: Consolas, monospace;")
        root.addWidget(self.leaderboard_label)

        human_row = QtWidgets.QHBoxLayout()
        root.addLayout(human_row)

        self.human_label = QtWidgets.QLabel("Human Player: disabled")
        human_row.addWidget(self.human_label)

        human_row.addSpacing(6)
        human_row.addWidget(QtWidgets.QLabel("Propose:"))
        self.human_propose_combo = QtWidgets.QComboBox()
        self.human_propose_combo.setMinimumWidth(120)
        human_row.addWidget(self.human_propose_combo)

        human_row.addWidget(QtWidgets.QLabel("Respond:"))
        self.human_response_combo = QtWidgets.QComboBox()
        self.human_response_combo.setMinimumWidth(120)
        human_row.addWidget(self.human_response_combo)

        human_row.addWidget(QtWidgets.QLabel("Play:"))
        self.human_play_combo = QtWidgets.QComboBox()
        self.human_play_combo.addItems(["Betray", "Befriend"])
        self.human_play_combo.setMinimumWidth(120)
        human_row.addWidget(self.human_play_combo)

        self.prepare_human_button = QtWidgets.QPushButton("Prepare Step")
        self.prepare_human_button.setEnabled(False)
        self.prepare_human_button.clicked.connect(self.on_prepare_human_step)
        human_row.addWidget(self.prepare_human_button)

        self.commit_human_button = QtWidgets.QPushButton("Commit Step")
        self.commit_human_button.setEnabled(False)
        self.commit_human_button.clicked.connect(self.on_commit_human_step)
        human_row.addWidget(self.commit_human_button)
        human_row.addStretch(1)

        self.figure = self.Figure(figsize=(14.5, 6.8), dpi=100)
        axes = self.figure.subplots(1, 3, gridspec_kw={"width_ratios": [1.05, 1.15, 1.2]})
        self.ax_graph, self.ax_heat, self.ax_metrics = axes
        self.ax_metrics_right = self.ax_metrics.twinx()
        self.canvas = self.FigureCanvasQTAgg(self.figure)
        self.toolbar = self.NavigationToolbar2QT(self.canvas, self.window)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=1)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(20_000)
        self.log_box.setMinimumHeight(220)
        root.addWidget(self.log_box)

        self.timer = QtCore.QTimer(self.window)
        self.timer.setInterval(max(1, int(self.args.gui_step_ms)))
        self.timer.timeout.connect(self._step_loop)

    def _bind_visualizer(self) -> None:
        if hasattr(self, "visualizer") and self.visualizer is not None and self.visualizer.heat_colorbar is not None:
            try:
                self.visualizer.heat_colorbar.remove()
            except Exception:
                pass
        self.visualizer = MiniSocietyVisualizer(
            n_agents=self.trainer.n_agents,
            human_agent_id=self.trainer.human_agent_id,
            cooperation_window=self.args.coop_window,
            reward_window=self.args.reward_window,
            pause_seconds=0.0,
        )
        self.visualizer.fig = self.figure
        self.visualizer.ax_graph = self.ax_graph
        self.visualizer.ax_heat = self.ax_heat
        self.visualizer.ax_metrics = self.ax_metrics
        self.visualizer.ax_metrics_right = self.ax_metrics_right
        self.visualizer.heat_colorbar = None
        self.visualizer._live_interactive_backend = False

    def _human_propose_values(self) -> List[str]:
        if not self.human_mode:
            return []
        assert self.trainer.human_agent_id is not None
        values = [f"A{i}" for i in range(self.trainer.n_agents) if i != self.trainer.human_agent_id]
        values.append("None")
        return values

    def _sync_human_controls_ui(self) -> None:
        if self.human_mode and self.trainer.human_agent_id is not None:
            self.human_label.setText(f"Human Player: A{self.trainer.human_agent_id}")
            values = self._human_propose_values()
            self.human_propose_combo.clear()
            self.human_propose_combo.addItems(values if len(values) > 0 else ["None"])
            if self.human_propose_combo.count() > 0:
                self.human_propose_combo.setCurrentIndex(max(0, self.human_propose_combo.count() - 1))
            self.human_response_combo.clear()
            self.human_response_combo.addItems(["Decline"])
            self.human_play_combo.setCurrentText("Befriend")
            self.human_propose_combo.setEnabled(True)
            self.human_response_combo.setEnabled(True)
            self.human_play_combo.setEnabled(True)
        else:
            self.human_label.setText("Human Player: disabled")
            self.human_propose_combo.clear()
            self.human_propose_combo.addItems(["None"])
            self.human_response_combo.clear()
            self.human_response_combo.addItems(["Decline"])
            self.human_play_combo.setCurrentText("Befriend")
            self.human_propose_combo.setEnabled(False)
            self.human_response_combo.setEnabled(False)
            self.human_play_combo.setEnabled(False)
        self.prepare_human_button.setEnabled(False)
        self.commit_human_button.setEnabled(False)

    def _clear_visualization(self) -> None:
        self.ax_graph.clear()
        self.ax_heat.clear()
        self.ax_metrics.clear()
        self.ax_metrics_right.clear()
        self.figure.suptitle("")
        self.canvas.draw_idle()

    def _reset_simulation_from_args(self, reason: str) -> None:
        if self.timer.isActive():
            self.timer.stop()
        set_seed(self.args.seed)
        self.payoff = build_payoff_from_args(self.args)
        self.trainer = build_trainer_from_args(self.args, self.payoff)
        self.human_mode = self.trainer.human_agent_id is not None
        self.continuous_mode = bool(self.args.continuous or self.args.episode_length == 0)
        self.total_episodes = self.args.episodes
        self.current_episode = 1
        self.global_steps = 0
        self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]
        self.tracker = None
        self.last_history = []
        self.pending_human_step = None
        self.running = False
        self.paused = False
        self.timer.setInterval(max(1, int(self.args.gui_step_ms)))
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.save_path_edit.setText(self.args.checkpoint_path)
        self._bind_visualizer()
        self._sync_human_controls_ui()
        self._clear_visualization()
        self._update_status()
        self._update_leaderboard()
        self.log(reason)

    def _parse_human_agent_choice(self, raw: str, decline_token: str) -> int:
        value = raw.strip()
        if value == decline_token:
            return self.trainer.n_agents
        if value.startswith("A"):
            return int(value[1:])
        raise ValueError(f"Invalid agent choice: {value}")

    @staticmethod
    def _parse_human_play_choice(raw: str) -> int:
        value = raw.strip().lower()
        if value == "betray":
            return 0
        if value == "befriend":
            return 1
        raise ValueError(f"Invalid play choice: {raw}")

    def _accumulate_global_scores(self, rewards: Sequence[float]) -> None:
        if len(self.global_scores) != self.trainer.n_agents:
            self.global_scores = [0.0 for _ in range(self.trainer.n_agents)]
        for i, value in enumerate(rewards):
            self.global_scores[i] += float(value)

    def _update_leaderboard(self) -> None:
        self.leaderboard_label.setText(
            format_gui_leaderboard(
                self.global_scores,
                human_mode=self.human_mode,
                human_agent_id=self.trainer.human_agent_id,
            )
        )

    def _update_status(self) -> None:
        phase = "running" if self.running and not self.paused else "paused" if self.paused else "idle"
        step_now = self.trainer.env.timestep
        episode_text = "continuous" if self.continuous_mode else f"{self.current_episode}/{self.total_episodes}"
        mode_text = "human" if self.human_mode else "ai-only"
        pending_text = "yes" if self.pending_human_step is not None else "no"
        self.status_label.setText(
            format_gui_status(
                phase=phase,
                mode_text=mode_text,
                episode_text=episode_text,
                step_now=step_now,
                episode_length=self.trainer.env.episode_length,
                global_steps=self.global_steps,
                pending_text=pending_text,
            )
        )

    def _get_history_source(self) -> List[Dict[str, object]]:
        if self.tracker is not None and len(self.tracker.step_history) > 0:
            return self.tracker.step_history
        return self.last_history

    def _render_if_available(self) -> None:
        history = self._get_history_source()
        if len(history) == 0:
            return
        step_idx = len(history)
        total_steps = self.trainer.env.episode_length if self.trainer.env.episode_length > 0 else step_idx
        step_data = history[-1]
        self.visualizer._draw_social_graph(step_data, step_idx=step_idx, total_steps=total_steps)
        self.visualizer._draw_trust_heatmap(step_data)
        self.visualizer._draw_metrics(history)
        self.visualizer._apply_layout("Mini-Society Dashboard (Live)")
        self.canvas.draw_idle()

    def log(self, message: str) -> None:
        self.log_box.appendPlainText(message)

    def _finish_current_episode(self) -> None:
        if self.tracker is None:
            return
        result = self.trainer.finish_episode(self.tracker, terminal=bool(self.tracker.done))
        if len(self.tracker.step_history) > 0:
            self.last_history = list(self.tracker.step_history)
        summary = format_episode_summary(self.current_episode, result)
        summary = summary.replace(
            f"ep={self.current_episode:04d}",
            format_run_label(self.current_episode, self.continuous_mode),
        )
        self.log(summary)
        self.tracker = None
        self.pending_human_step = None
        if self.human_mode:
            self.commit_human_button.setEnabled(False)
            self.human_response_combo.clear()
            self.human_response_combo.addItems(["Decline"])
        if not self.continuous_mode:
            self.current_episode += 1

    def _finalize_completed_run(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.running = False
        self.paused = False
        self.pending_human_step = None
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.start_button.setEnabled(True)
        self.prepare_human_button.setEnabled(False)
        self.commit_human_button.setEnabled(False)
        self._update_status()
        self._update_leaderboard()
        self.log("Run completed.")

        if self.args.save_checkpoint.strip():
            save_checkpoint(
                path=self.args.save_checkpoint.strip(),
                trainer=self.trainer,
                current_episode=self.current_episode,
                total_episodes=self.total_episodes,
                tracker=self.tracker,
                global_steps=self.global_steps,
                global_scores=self.global_scores,
            )
            self.log(f"Saved checkpoint: {self.args.save_checkpoint.strip()}")

        if self.args.viz_output.strip() and len(self.last_history) > 0:
            self.figure.savefig(self.args.viz_output.strip(), dpi=170, bbox_inches="tight")
            self.log(f"Saved visualization snapshot: {self.args.viz_output.strip()}")


    def _step_loop(self) -> None:
        if not self.running or self.paused:
            return
        if self.human_mode:
            if self.timer.isActive():
                self.timer.stop()
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return
            if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
                self.log(f"Reached max global steps: {self.args.max_steps}")
                self._finalize_completed_run()
                return
            if self.tracker is None:
                self.tracker = self.trainer.start_episode(collect_history=True)
            self.prepare_human_button.setEnabled(True)
            self.commit_human_button.setEnabled(self.pending_human_step is not None)
            self._update_status()
            return

        if not self.continuous_mode and self.current_episode > self.total_episodes:
            self._finalize_completed_run()
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)

        info = self.trainer.step_episode(self.tracker, collect_history=True)
        self._accumulate_global_scores(info["agent_rewards"])  # type: ignore[index]
        self.global_steps += 1
        self._update_leaderboard()
        self._render_if_available()
        self._update_status()

        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            if self.tracker is not None:
                self.tracker.done = False
                self._finish_current_episode()
            self.log(f"Reached max global steps: {self.args.max_steps}")
            self._finalize_completed_run()
            return

        if self.tracker.done:
            self._finish_current_episode()
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return

    def on_start(self) -> None:
        if not self.continuous_mode and self.current_episode > self.total_episodes:
            self.log("All episodes are already complete. Click Restart to begin again.")
            return
        if self.running and not self.paused:
            return
        self.running = True
        self.paused = False
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self._update_status()
        if self.human_mode:
            self._step_loop()
            return
        if not self.timer.isActive():
            self.timer.start()

    def on_pause_resume(self) -> None:
        if not self.running:
            self.on_start()
            return
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText("Resume")
            if self.timer.isActive():
                self.timer.stop()
            self.prepare_human_button.setEnabled(False)
            self.commit_human_button.setEnabled(False)
        else:
            self.pause_button.setText("Pause")
            if self.human_mode:
                self._step_loop()
            else:
                if not self.timer.isActive():
                    self.timer.start()
        self._update_status()

    def on_status(self) -> None:
        self.log(format_agent_state_snapshot(self.trainer))
        self.log(self.leaderboard_label.text())

    def on_history(self) -> None:
        tail = int(self.history_k_spin.value())
        self.log(format_recent_interactions(self._get_history_source(), tail=tail))

    def on_prepare_human_step(self) -> None:
        if not self.human_mode:
            self.log("Human controls are unavailable in AI-only mode.")
            return
        if not self.running or self.paused:
            self.log("Start or resume the simulation before preparing a human step.")
            return
        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            self.log(f"Global step cap reached: {self.args.max_steps}")
            self._finalize_completed_run()
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)
        if self.tracker.done:
            self._finish_current_episode()
            self._update_status()
            return

        try:
            human_propose = self._parse_human_agent_choice(self.human_propose_combo.currentText(), decline_token="None")
            prepared = self.trainer.prepare_human_step(human_propose_action=human_propose)
        except Exception as exc:
            self.log(f"Failed to prepare human step: {exc}")
            return

        self.pending_human_step = prepared
        assert self.trainer.human_agent_id is not None
        human_id = self.trainer.human_agent_id
        mask = prepared.respond_masks[human_id].detach().cpu().tolist()
        response_values = [f"A{i}" for i in range(self.trainer.n_agents) if bool(mask[i])]
        response_values.append("Decline")
        self.human_response_combo.clear()
        self.human_response_combo.addItems(response_values)
        self.human_response_combo.setCurrentText("Decline")
        self.commit_human_button.setEnabled(True)

        incoming = [
            i
            for i, v in enumerate(self.trainer.env.states[human_id].incoming_proposals.detach().cpu().tolist())
            if int(v) == 1
        ]
        incoming_text = "none" if len(incoming) == 0 else ", ".join(f"A{i}" for i in incoming)
        self.log(
            f"Prepared step t={prepared.timestep}: human propose={self.human_propose_combo.currentText()} "
            f"| incoming proposals -> {incoming_text}"
        )
        self._update_status()


    def on_commit_human_step(self) -> None:
        if not self.human_mode:
            self.log("Human controls are unavailable in AI-only mode.")
            return
        if not self.running or self.paused:
            self.log("Start or resume the simulation before committing a human step.")
            return
        if self.pending_human_step is None:
            self.log("No prepared human step. Click 'Prepare Step' first.")
            return
        if self.tracker is None:
            self.tracker = self.trainer.start_episode(collect_history=True)

        try:
            human_response = self._parse_human_agent_choice(
                self.human_response_combo.currentText(), decline_token="Decline"
            )
            human_play = self._parse_human_play_choice(self.human_play_combo.currentText())
            info = self.trainer.execute_prepared_human_step(
                tracker=self.tracker,
                prepared=self.pending_human_step,
                human_response_action=human_response,
                human_play_action=human_play,
                collect_history=True,
            )
        except Exception as exc:
            self.log(f"Failed to commit human step: {exc}")
            return

        self.pending_human_step = None
        self.commit_human_button.setEnabled(False)
        self._accumulate_global_scores(info["agent_rewards"])  # type: ignore[index]
        self.global_steps += 1
        self._update_leaderboard()
        self._render_if_available()
        self._update_status()

        if self.args.max_steps > 0 and self.global_steps >= self.args.max_steps:
            if self.tracker is not None:
                self.tracker.done = False
                self._finish_current_episode()
            self.log(f"Reached max global steps: {self.args.max_steps}")
            self._finalize_completed_run()
            return

        if self.tracker is not None and self.tracker.done:
            self._finish_current_episode()
            if not self.continuous_mode and self.current_episode > self.total_episodes:
                self._finalize_completed_run()
                return

    def on_save(self) -> None:
        path = self.save_path_edit.text().strip() or self.args.checkpoint_path
        if not path:
            self.log("No save path provided.")
            return
        save_checkpoint(
            path=path,
            trainer=self.trainer,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            tracker=self.tracker,
            global_steps=self.global_steps,
            global_scores=self.global_scores,
        )
        self.log(f"Checkpoint saved: {path}")

    def _ask_yes_no(self, title: str, text: str) -> bool:
        QMessageBox = self.QtWidgets.QMessageBox
        answer = QMessageBox.question(
            self.window,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def on_restart(self) -> None:
        if not self._ask_yes_no("Restart", "Restart from scratch? Current unsaved progress will be lost."):
            return
        self._reset_simulation_from_args("Simulation restarted from scratch.")

    def on_help(self) -> None:
        self.log("start: begin simulation")
        self.log("pause/resume: pause after current step or continue")
        self.log("continuous mode: set continuous=true in Hyperparams")
        self.log("leaderboard: cumulative scores across all elapsed steps")
        if self.human_mode:
            self.log("human mode: choose propose/respond/play each step")
            self.log("prepare step: samples AI proposals/responses and updates valid response options")
            self.log("commit step: executes one full environment step with selected response/play")
        self.log("status: print current trust/confidence/incoming proposals for all agents")
        self.log("history K: show latest K interaction steps")
        self.log("save: write full checkpoint to save path")
        self.log("hyperparams: open editor to view/change all args, then apply + restart")
        self.log("restart: reset env and agents from scratch")
        self.log("quit: close app")

    def on_open_hyperparams(self) -> None:
        QtWidgets = self.QtWidgets
        if self.hyperparams_dialog is not None and self.hyperparams_dialog.isVisible():
            self.hyperparams_dialog.raise_()
            self.hyperparams_dialog.activateWindow()
            return

        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle("Hyperparameter Editor")
        dialog.resize(760, 760)
        self.hyperparams_dialog = dialog
        self.hyperparam_widgets = {}

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel("Edit values, then Apply + Restart. Lists use comma-separated values.")
        )

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_host)
        scroll.setWidget(form_host)
        layout.addWidget(scroll, stretch=1)

        args_dict = vars(self.args)
        for key, value in args_dict.items():
            if isinstance(value, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(bool(value))
                meta: Dict[str, object] = {"kind": "bool", "widget": widget}
            else:
                if isinstance(value, list):
                    text = ", ".join(str(v) for v in value)
                    meta = {
                        "kind": "list",
                        "widget": QtWidgets.QLineEdit(text),
                        "list_len": len(value),
                    }
                else:
                    meta = {
                        "kind": "scalar",
                        "widget": QtWidgets.QLineEdit(str(value)),
                        "py_type": type(value),
                    }
            widget_obj = meta["widget"]
            if key == "gui":
                widget_obj.setEnabled(False)  # type: ignore[union-attr]
            form.addRow(QtWidgets.QLabel(key), widget_obj)  # type: ignore[arg-type]
            self.hyperparam_widgets[key] = meta

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        apply_btn = QtWidgets.QPushButton("Apply + Restart")
        reload_btn = QtWidgets.QPushButton("Reload Current")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reload_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        apply_btn.clicked.connect(self.on_apply_hyperparams)
        reload_btn.clicked.connect(self.on_reload_hyperparams)
        close_btn.clicked.connect(dialog.close)
        dialog.destroyed.connect(self._on_hyperparams_closed)
        dialog.show()

    def _on_hyperparams_closed(self, *_args: object) -> None:
        self.hyperparams_dialog = None

    def on_reload_hyperparams(self) -> None:
        if self.hyperparams_dialog is not None:
            self.hyperparams_dialog.close()
        self.on_open_hyperparams()

    def on_apply_hyperparams(self) -> None:
        if self.hyperparams_dialog is None:
            return

        new_args = argparse.Namespace(**dict(vars(self.args)))
        try:
            for key, meta in self.hyperparam_widgets.items():
                kind = str(meta["kind"])
                widget = meta["widget"]
                if kind == "bool":
                    setattr(new_args, key, bool(widget.isChecked()))
                    continue
                text = str(widget.text()).strip()
                if kind == "list":
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    expected_len = int(meta["list_len"])
                    if len(parts) != expected_len:
                        raise ValueError(f"{key} expects {expected_len} comma-separated values.")
                    setattr(new_args, key, [float(p) for p in parts])
                    continue
                py_type = meta["py_type"]
                if py_type is int:
                    setattr(new_args, key, int(text))
                elif py_type is float:
                    setattr(new_args, key, float(text))
                else:
                    setattr(new_args, key, text)
        except ValueError as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Invalid Hyperparameter", str(exc))
            return

        new_args.gui = True
        if bool(new_args.continuous):
            new_args.episode_length = 0

        try:
            validate_args(new_args)
            build_payoff_from_args(new_args)
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Validation Error", str(exc))
            return

        if self.running or self.tracker is not None:
            if not self._ask_yes_no("Apply Hyperparameters", "Apply new hyperparameters and restart simulation now?"):
                return

        self.args = new_args
        try:
            self._reset_simulation_from_args("Applied hyperparameters from UI and restarted simulation.")
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Apply Failed", str(exc))
            return

        if self.hyperparams_dialog is not None:
            self.hyperparams_dialog.close()
            self.hyperparams_dialog = None

    def on_quit(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.running = False
        self.window.close()

    def _on_close_event(self, event: object) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.running = False
        self.paused = False
        if self.hyperparams_dialog is not None:
            self.hyperparams_dialog.close()
        event.accept()  # type: ignore[attr-defined]

    def run(self) -> None:
        self.window.show()
        self.app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decentralized MARL mini-society prototype (PyTorch PPO).")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument(
        "--human-player",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable human-controlled mode for one agent (intended for GUI control panel).",
    )
    parser.add_argument(
        "--human-agent-id",
        type=int,
        default=0,
        help="Agent index controlled by the human when --human-player is enabled.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=64,
        help="Per-episode step limit (0 = unbounded horizon; agents never observe this value).",
    )
    parser.add_argument("--confidence-decay", type=float, default=0.005)
    parser.add_argument("--gossip-noise", type=float, default=0.04)
    parser.add_argument("--gossip-strength", type=float, default=0.18)
    parser.add_argument("--initial-confidence", type=float, default=0.15)
    parser.add_argument("--rep-delta-befriend", type=float, default=0.15)
    parser.add_argument("--rep-delta-betray", type=float, default=-0.35)
    parser.add_argument("--conf-gain-interaction", type=float, default=1.0)
    parser.add_argument(
        "--payoff-mutual-befriend",
        nargs=2,
        type=float,
        metavar=("A", "B"),
        default=[100.0, 100.0],
        help="Reward tuple for (befriend, befriend).",
    )
    parser.add_argument(
        "--payoff-betray-befriend",
        nargs=2,
        type=float,
        metavar=("A", "B"),
        default=[103.0, -106.0],
        help="Reward tuple for (betray, befriend).",
    )
    parser.add_argument(
        "--payoff-befriend-betray",
        nargs=2,
        type=float,
        metavar=("A", "B"),
        default=[-106.0, 103.0],
        help="Reward tuple for (befriend, betray).",
    )
    parser.add_argument(
        "--payoff-mutual-betray",
        nargs=2,
        type=float,
        metavar=("A", "B"),
        default=[-2.0, -2.0],
        help="Reward tuple for (betray, betray).",
    )
    parser.add_argument("--isolation-penalty", type=float, default=-1.0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for PPO advantage estimation (1.0 approaches Monte Carlo returns).",
    )
    parser.add_argument(
        "--continuous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run as one continuous simulation trajectory (no episode resets).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="Number of steps per PPO rollout update chunk (used even when horizon is unbounded).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Global run cap in environment steps (0 disables cap). Agents do not observe this limit.",
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch PySide6 dashboard with button-based controls and live visualizations.",
    )
    parser.add_argument(
        "--gui-step-ms",
        type=int,
        default=150,
        help="Delay between simulation steps in GUI mode (milliseconds).",
    )
    parser.add_argument(
        "--interactive-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable runtime pause/resume/restart controls (press 'p' in terminal to pause after a step).",
    )
    parser.add_argument(
        "--pause-poll-interval",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) per step in interactive mode. Useful to slow simulation for manual control.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="minisociety_checkpoint.pt",
        help="Default checkpoint path used by pause-menu save command.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default="",
        help="If set, save a checkpoint at the end of the run to this path.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to load before running.",
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        default="full",
        choices=["full", "weights_only"],
        help="Checkpoint load mode: full resume or weights_only with fresh env.",
    )
    parser.add_argument(
        "--transfer-trust-priors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using --load-mode weights_only, seed fresh env with checkpoint trust priors.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Animate the final episode using a 3-panel dashboard.",
    )
    parser.add_argument("--viz-pause", type=float, default=0.2, help="Pause in seconds between rendered steps.")
    parser.add_argument(
        "--viz-output",
        type=str,
        default="",
        help="Optional output image path for final visualization frame (useful for headless runs).",
    )
    parser.add_argument("--coop-window", type=int, default=10, help="Rolling window for cooperation rate.")
    parser.add_argument("--reward-window", type=int, default=10, help="Rolling window for societal reward average.")
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable reward normalization before computing returns.",
    )
    parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable advantage normalization.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    validate_args(args)
    if args.continuous:
        args.episode_length = 0

    payoff = build_payoff_from_args(args)
    set_seed(args.seed)

    if args.gui:
        app = MiniSocietyQtApp(args=args, payoff=payoff)
        app.on_help()
        app.run()
        return

    trainer = build_trainer_from_args(args, payoff)
    current_episode = 1
    global_steps = 0
    global_scores = [0.0 for _ in range(trainer.n_agents)]
    tracker: Optional[EpisodeTracker] = None
    if args.load_checkpoint.strip():
        trainer, current_episode, tracker, global_steps, loaded_scores = load_into_trainer(
            args=args,
            trainer=trainer,
            checkpoint_path=args.load_checkpoint.strip(),
        )
        global_scores = loaded_scores if loaded_scores is not None else [0.0 for _ in range(trainer.n_agents)]
    if trainer.human_agent_id is not None:
        raise ValueError(
            "Human-player mode is active but CLI non-GUI execution cannot collect human actions. "
            "Launch with --gui."
        )

    visualizer: Optional[MiniSocietyVisualizer] = None
    if args.visualize:
        visualizer = MiniSocietyVisualizer(
            n_agents=trainer.n_agents,
            human_agent_id=trainer.human_agent_id,
            cooperation_window=args.coop_window,
            reward_window=args.reward_window,
            pause_seconds=args.viz_pause,
        )

    pause_watcher = ConsolePauseWatcher() if args.interactive_controls else None
    if args.interactive_controls:
        print("Interactive mode enabled. Press 'p' in terminal to pause after the current step.")
        if not pause_watcher or not pause_watcher.enabled:
            print(
                "Non-blocking key polling is unavailable on this platform; "
                "pause using a tiny --pause-poll-interval and Ctrl+C if needed."
            )

    last_history: Optional[List[Dict[str, object]]] = None
    quit_requested = False
    stop_due_step_cap = False
    continuous_mode = bool(args.continuous or args.episode_length == 0)

    while (continuous_mode or current_episode <= args.episodes) and not quit_requested:
        collect_history = bool(
            args.interactive_controls
            or (args.visualize and (args.interactive_controls or continuous_mode or current_episode == args.episodes))
        )
        if tracker is None:
            tracker = trainer.start_episode(collect_history=collect_history)

        restart_requested = False
        while tracker is not None and not tracker.done:
            info = trainer.step_episode(tracker, collect_history=collect_history)
            for i, reward in enumerate(info["agent_rewards"]):  # type: ignore[index]
                global_scores[i] += float(reward)
            global_steps += 1
            if args.visualize and args.interactive_controls and visualizer is not None:
                visualizer.render_step(tracker.step_history)
            if args.pause_poll_interval > 0:
                time.sleep(args.pause_poll_interval)

            if args.max_steps > 0 and global_steps >= args.max_steps:
                stop_due_step_cap = True
                break

            if args.interactive_controls and pause_watcher is not None and pause_watcher.poll_pause_requested():
                action = pause_menu(
                    trainer=trainer,
                    tracker=tracker,
                    current_episode=current_episode,
                    total_episodes=args.episodes,
                    default_checkpoint_path=args.checkpoint_path,
                    global_steps=global_steps,
                    global_scores=global_scores,
                )
                if action == "quit":
                    quit_requested = True
                    break
                if action == "restart":
                    set_seed(args.seed)
                    trainer = build_trainer_from_args(args, payoff)
                    tracker = None
                    current_episode = 1
                    global_steps = 0
                    global_scores = [0.0 for _ in range(trainer.n_agents)]
                    restart_requested = True
                    break

        if quit_requested:
            break
        if restart_requested:
            continue
        if tracker is None:
            continue

        terminal_end = bool(tracker.done)
        if stop_due_step_cap and not tracker.done:
            terminal_end = False
        result = trainer.finish_episode(tracker, terminal=terminal_end)
        if len(tracker.step_history) > 0:
            last_history = tracker.step_history

        if (
            stop_due_step_cap
            or current_episode == 1
            or current_episode % args.print_every == 0
            or current_episode == args.episodes
        ):
            summary = format_episode_summary(current_episode, result)
            summary = summary.replace(f"ep={current_episode:04d}", format_run_label(current_episode, continuous_mode))
            print(summary)

        if stop_due_step_cap:
            print(f"Reached max global steps: {args.max_steps}")
            tracker = None
            break

        tracker = None
        if not continuous_mode:
            current_episode += 1

    if args.save_checkpoint.strip():
        save_checkpoint(
            path=args.save_checkpoint.strip(),
            trainer=trainer,
            current_episode=current_episode,
            total_episodes=args.episodes,
            tracker=tracker,
            global_steps=global_steps,
            global_scores=global_scores,
        )
        print(f"Saved checkpoint: {args.save_checkpoint.strip()}")

    if args.visualize and visualizer is not None:
        output_path = args.viz_output.strip() or None
        if args.interactive_controls:
            visualizer.finalize_live(output_path=output_path)
        elif last_history is not None:
            visualizer.animate_episode(last_history, output_path=output_path)


if __name__ == "__main__":
    main()
