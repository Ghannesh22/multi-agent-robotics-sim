from __future__ import annotations

import importlib.util
import unittest

from marlsim.physics import PhysicsWorld


class PhysicsWorldTests(unittest.TestCase):
    def test_world_starts_disconnected(self) -> None:
        world = PhysicsWorld()

        self.assertFalse(world.is_connected)
        self.assertIsNone(world.client_id)
        self.assertIsNone(world.plane_id)

    def test_step_requires_connection(self) -> None:
        world = PhysicsWorld()

        with self.assertRaises(RuntimeError):
            world.step()

    @unittest.skipIf(
        importlib.util.find_spec("pybullet") is None,
        "PyBullet optional dependency is not installed.",
    )
    def test_direct_world_creates_plane_and_steps(self) -> None:
        world = PhysicsWorld(mode="DIRECT")

        try:
            client_id = world.connect()
            plane_id = world.create_plane()
            world.step(steps=3)

            self.assertIsInstance(client_id, int)
            self.assertIsInstance(plane_id, int)
            self.assertTrue(world.is_connected)
        finally:
            world.disconnect()

        self.assertFalse(world.is_connected)
        self.assertIsNone(world.plane_id)
