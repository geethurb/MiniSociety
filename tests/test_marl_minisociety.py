import math
import tempfile
import unittest
from pathlib import Path

import marl_minisociety as ms


class MiniSocietyTests(unittest.TestCase):
    def test_default_exploit_outcome_is_net_negative(self) -> None:
        payoff = ms.PayoffConfig()
        self.assertGreater(payoff.betray_befriend[0], payoff.mutual_befriend[0])
        self.assertLess(sum(payoff.betray_befriend), 0.0)
        self.assertLess(sum(payoff.betray_befriend), 2.0 * payoff.isolation)
        self.assertLess(payoff.mutual_betray[0], payoff.isolation)
        self.assertLess(payoff.mutual_betray[1], payoff.isolation)

    def test_gossip_preserves_direct_partner_memory(self) -> None:
        env = ms.MiniSocietyEnv(
            n_agents=4,
            confidence_decay=0.0,
            gossip_noise=0.0,
            gossip_strength=0.5,
            device="cpu",
        )
        env.states[0].trust_matrix[1] = env.states[0].trust_matrix[1].new_tensor([-0.6, 0.9])
        env.states[1].trust_matrix[0] = env.states[1].trust_matrix[0].new_tensor([0.4, 0.8])
        env.states[0].trust_matrix[2] = env.states[0].trust_matrix[2].new_tensor([0.8, 0.6])
        env.states[1].trust_matrix[2] = env.states[1].trust_matrix[2].new_tensor([-0.2, 0.2])

        direct_a = env.states[0].trust_matrix[1].clone()
        direct_b = env.states[1].trust_matrix[0].clone()
        env._gossip_blend(0, 1)

        self.assertTrue(env.states[0].trust_matrix[1].equal(direct_a))
        self.assertTrue(env.states[1].trust_matrix[0].equal(direct_b))
        self.assertAlmostEqual(float(env.states[0].trust_matrix[2, 0]), 0.6571429, places=5)
        self.assertAlmostEqual(float(env.states[1].trust_matrix[2, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(env.states[0].trust_matrix[2, 1]), 0.64, places=5)
        self.assertAlmostEqual(float(env.states[1].trust_matrix[2, 1]), 0.44, places=5)

    def test_direct_interaction_sets_full_confidence_after_step(self) -> None:
        env = ms.MiniSocietyEnv(
            n_agents=2,
            confidence_decay=0.5,
            gossip_noise=0.0,
            gossip_strength=0.0,
            initial_confidence=0.2,
            confidence_gain_interaction=1.0,
            device="cpu",
        )

        env._resolve_step(
            pairs=[(0, 1)],
            proposals=[1, 0],
            responses=[1, 0],
            play_actions={0: 1, 1: 0},
        )

        self.assertAlmostEqual(float(env.states[0].trust_matrix[1, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(env.states[1].trust_matrix[0, 1]), 1.0, places=6)

    def test_interaction_step_applies_direct_and_partner_trust_updates(self) -> None:
        env = ms.MiniSocietyEnv(
            n_agents=3,
            confidence_decay=0.0,
            gossip_noise=0.0,
            gossip_strength=0.5,
            initial_confidence=0.1,
            confidence_gain_interaction=1.0,
            device="cpu",
        )
        env.states[0].trust_matrix[2] = env.states[0].trust_matrix[2].new_tensor([-0.4, 0.1])
        env.states[1].trust_matrix[2] = env.states[1].trust_matrix[2].new_tensor([0.8, 0.9])

        env._resolve_step(
            pairs=[(0, 1)],
            proposals=[1, 0, 3],
            responses=[1, 0, 3],
            play_actions={0: 1, 1: 0},
        )

        # Direct interaction with agent 1 is the strongest evidence.
        self.assertAlmostEqual(float(env.states[0].trust_matrix[1, 0]), -0.35, places=6)
        self.assertAlmostEqual(float(env.states[0].trust_matrix[1, 1]), 1.0, places=6)

        # Agent 1's opinion of agent 2 is also incorporated, but with lower confidence.
        self.assertGreater(float(env.states[0].trust_matrix[2, 0]), -0.4)
        self.assertLess(float(env.states[0].trust_matrix[2, 0]), 0.8)
        self.assertGreater(float(env.states[0].trust_matrix[2, 1]), 0.1)
        self.assertLess(float(env.states[0].trust_matrix[2, 1]), float(env.states[0].trust_matrix[1, 1]))

    def test_observation_includes_targets_view_of_observer(self) -> None:
        env = ms.MiniSocietyEnv(n_agents=3, device="cpu")
        env.states[0].trust_matrix[1] = env.states[0].trust_matrix[1].new_tensor([0.25, 0.6])
        env.states[1].trust_matrix[0] = env.states[1].trust_matrix[0].new_tensor([-0.4, 0.9])
        env.states[2].trust_matrix[0] = env.states[2].trust_matrix[0].new_tensor([0.7, 0.3])
        env.states[0].incoming_proposals[2] = 1.0

        obs = env.observe(0)

        self.assertEqual(obs.shape[0], env.n_agents * 5)
        reciprocal = obs[env.n_agents * 3 :].reshape(env.n_agents, 2)
        self.assertAlmostEqual(float(reciprocal[1, 0]), -0.4, places=6)
        self.assertAlmostEqual(float(reciprocal[1, 1]), 0.9, places=6)
        self.assertAlmostEqual(float(reciprocal[2, 0]), 0.7, places=6)
        self.assertAlmostEqual(float(reciprocal[2, 1]), 0.3, places=6)

    def test_run_episode_smoke_with_gae(self) -> None:
        ms.set_seed(7)
        trainer = ms.MiniSocietyTrainer(
            n_agents=4,
            episode_length=6,
            rollout_steps=3,
            gae_lambda=0.95,
            device="cpu",
        )

        result = trainer.run_episode(collect_history=True)

        self.assertEqual(result["steps"], 6)
        self.assertEqual(len(result["step_history"]), 6)
        self.assertGreater(result["updates"], 0)
        self.assertTrue(math.isfinite(float(result["mean_loss"])))
        self.assertTrue(math.isfinite(float(result["mean_entropy"])))

    def test_human_step_prepare_and_commit(self) -> None:
        ms.set_seed(11)
        trainer = ms.MiniSocietyTrainer(
            n_agents=4,
            episode_length=4,
            rollout_steps=2,
            gae_lambda=0.95,
            human_agent_id=0,
            device="cpu",
        )
        tracker = trainer.start_episode(collect_history=True)

        prepared = trainer.prepare_human_step(human_propose_action=trainer.n_agents)
        info = trainer.execute_prepared_human_step(
            tracker=tracker,
            prepared=prepared,
            human_response_action=trainer.n_agents,
            human_play_action=1,
            collect_history=True,
        )

        self.assertEqual(tracker.step_count, 1)
        self.assertEqual(len(tracker.step_history), 1)
        self.assertEqual(len(info["agent_rewards"]), trainer.n_agents)

    def test_checkpoint_roundtrip_creates_parent_dir(self) -> None:
        ms.set_seed(19)
        trainer = ms.MiniSocietyTrainer(
            n_agents=4,
            episode_length=5,
            rollout_steps=2,
            gae_lambda=0.95,
            device="cpu",
        )
        trainer.run_episode(collect_history=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "nested" / "checkpoints" / "society.pt"
            ms.save_checkpoint(
                path=str(checkpoint_path),
                trainer=trainer,
                current_episode=2,
                total_episodes=5,
                tracker=None,
                global_steps=trainer.env.timestep,
                global_scores=[1.0 for _ in range(trainer.n_agents)],
            )

            self.assertTrue(checkpoint_path.exists())
            payload = ms.load_checkpoint(str(checkpoint_path), device="cpu")
            trainer_state = payload["trainer_state"]
            restored = ms.MiniSocietyTrainer.from_config(trainer_state["config"])
            restored.load_state(trainer_state)

            self.assertEqual(restored.env.timestep, trainer.env.timestep)
            self.assertEqual(len(restored.agents), len(trainer.agents))


if __name__ == "__main__":
    unittest.main()
