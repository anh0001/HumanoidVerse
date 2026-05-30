import sys
import os
import math
from loguru import logger
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
# from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.timer import Timer

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab.terrains import TerrainGeneratorCfg
import omni.isaac.lab.terrains as terrain_gen

from omni.isaac.lab_assets import H1_CFG
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.envs import ViewerCfg

import omni.isaac.lab.sim as sim_utils

from humanoidverse.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
import builtins
import inspect
import copy
from humanoidverse.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG

from humanoidverse.simulator.isaacsim.event_cfg import EventCfg

from omni.isaac.lab.managers import EventManager

from omni.isaac.lab.managers import EventTermCfg as EventTerm

from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.envs.mdp as mdp
from humanoidverse.simulator.isaacsim.events import randomize_body_com
from humanoidverse.simulator.isaacsim.furrow_terrain import HfFurrowsTerrainCfg

class IsaacSim(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.device = device  # Add this line for compatibility
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand
        
        sim_config: SimulationCfg = SimulationCfg(dt=1./self.simulator_config.sim.fps, 
                                           render_interval=self.simulator_config.sim.render_interval, 
                                           device=self.sim_device,
                                           physx=PhysxCfg(bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                                                          solver_type=self.simulator_config.sim.physx.solver_type,
                                                          max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                                                          max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations))
        
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        
        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1./self.simulator_config.sim.fps}")
        logger.info(f"\tRendering step-size   : {1./self.simulator_config.sim.fps * self.simulator_config.sim.substeps}")


        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)
        
        
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=self.simulator_config.scene.num_envs, env_spacing=self.simulator_config.scene.env_spacing, replicate_physics=self.simulator_config.scene.replicate_physics)
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        
    
        
        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
        
        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu")


        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(self.domain_rand_config["friction_range"]),
                    "operation": "scale",
                },
            )

        if self.domain_rand_config.get("randomize_base_com", False):
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            self.robot_config.torso_name,
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["y"][0], self.domain_rand_config["base_com_range"]["z"][0]]),
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][1], self.domain_rand_config["base_com_range"]["y"][1], self.domain_rand_config["base_com_range"]["z"][1]])
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )  

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
        
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
                
        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        
        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None


        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation


        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # print the environment information
        logger.info("Completed setting up the environment...")
        
        
    def _setup_scene(self):
        # actuators = {
        #     "legs": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
        #         effort_limit={
        #             ".*_hip_yaw": 200.0,
        #             ".*_hip_roll": 200.0,
        #             ".*_hip_pitch": 200.0,
        #             ".*_knee": 300.0,
        #             "torso": 200.0,
        #         },
        #         velocity_limit={
        #             ".*_hip_yaw": 23.0,
        #             ".*_hip_roll": 23.0,
        #             ".*_hip_pitch": 23.0,
        #             ".*_knee": 14.0,
        #             "torso": 23.0,
        #         },
        #         stiffness=0,
        #         damping=0,
        #     ),
        #     "feet": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_ankle"],
        #         effort_limit=40,
        #         velocity_limit=9.0,
        #         stiffness=0,
        #         damping=0,
        #     ),
        #     "arms": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
        #         effort_limit={
        #             ".*_shoulder_pitch": 40.0,
        #             ".*_shoulder_roll": 40.0,
        #             ".*_shoulder_yaw": 18.0,
        #             ".*_elbow": 18.0,
        #         },
        #         velocity_limit={
        #             ".*_shoulder_pitch": 9.0,
        #             ".*_shoulder_roll": 9.0,
        #             ".*_shoulder_yaw": 20.0,
        #             ".*_elbow": 20.0,
        #         },
        #         stiffness=0,
        #         damping=0,
        #     ),
        # }
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file
        # prapare to override the spawn configuration in RoboVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(asset_root, asset_path),
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        )
        
        # prepare to override the articulation configuration in RoboVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )
       
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")    
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping
        
        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")


        # ImplicitActuatorCfg IdealPDActuatorCfg
        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            ) for i in range(len(dof_names_list))
        }
        # import ipdb; ipdb.set_trace()
        # actuators = {
        #     dof_names_list[i]: ImplicitActuatorCfg(
        #         joint_names_expr=[dof_names_list[i]],
        #         effort_limit=dof_effort_limit_list[i],
        #         velocity_limit=dof_vel_limit_list[i],
        #         stiffness=kp_list[i],
        #         damping=kd_list[i],
        #     ) for i in range(len(dof_names_list))
        # }

        # actuators={
        # "legs": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_hip_yaw_joint": 150.0,
        #         ".*_hip_roll_joint": 150.0,
        #         ".*_hip_pitch_joint": 200.0,
        #         ".*_knee_joint": 200.0,
        #         "torso_joint": 200.0,
        #     },
        #     damping={
        #         ".*_hip_yaw_joint": 5.0,
        #         ".*_hip_roll_joint": 5.0,
        #         ".*_hip_pitch_joint": 5.0,
        #         ".*_knee_joint": 5.0,
        #         "torso_joint": 5.0,
        #     },
        # ),
        # "feet": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_ankle_joint"],
        #     effort_limit=100,
        #     velocity_limit=100.0,
        #     stiffness={".*_ankle_joint": 20.0},
        #     damping={".*_ankle_joint": 4.0},
        # ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 40.0,
        #         ".*_shoulder_roll_joint": 40.0,
        #         ".*_shoulder_yaw_joint": 40.0,
        #         ".*_elbow_joint": 40.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch_joint": 10.0,
        #         ".*_shoulder_roll_joint": 10.0,
        #         ".*_shoulder_yaw_joint": 10.0,
        #         ".*_elbow_joint": 10.0,
        #     },
        # )
        # }
        
        # import ipdb; ipdb.set_trace()
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)
        # robot_articulation_config: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )

        # Add a height scanner to the torso to detect the height of the terrain mesh
        height_scanner_config = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        if (self.terrain_config.mesh_type == "heightfield") or (self.terrain_config.mesh_type == "trimesh"):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                        )

            # If a single selected terrain is requested with custom kwargs, approximate it here.
            # Currently supports a Perlin-like micro-roughness via uniform height noise.
            try:
                if getattr(self.terrain_config, "selected", False) and getattr(self.terrain_config, "terrain_kwargs", None):
                    tkwargs = self.terrain_config.terrain_kwargs
                    ttype = str(tkwargs.get("type", "")).lower()
                    if ttype == "perlin":
                        amp = float(tkwargs.get("amplitude", 0.0))
                        freq = float(tkwargs.get("frequency", 10.0))
                        # Map Perlin amplitude/frequency to a uniform noise config as an approximation.
                        # noise_range ~ [0.5*amp, amp]; noise_step ~ 1/freq (meters)
                        noise_low = max(0.0, 0.5 * amp)
                        noise_high = max(noise_low, amp)
                        noise_step = max(0.005, 1.0 / max(1e-6, freq))
                        # sub_terrains = {
                        #     "flat": terrain_gen.HfRandomUniformTerrainCfg(
                        #         proportion=1.0,
                        #         noise_range=(noise_low, noise_high),
                        #         noise_step=noise_step,
                        #         border_width=0.0,
                        #     )
                        # }
                    elif ttype == "furrows":
                        # Map our YAML kwargs to a custom HF terrain that draws parallel grooves.
                        # Accept both the legacy "*_m"/"orientation_deg" keys (used in gym utils)
                        # and the Isaac Lab-native keys used by HfFurrowsTerrainCfg.
                        depth_rng = (
                            tkwargs.get("depth_range")
                            or tkwargs.get("depth_range_m")
                            or [0.05, 0.15]
                        )
                        spacing_rng = (
                            tkwargs.get("spacing_range")
                            or tkwargs.get("spacing_range_m")
                            or [0.8, 1.2]
                        )
                        orient_rng = (
                            tkwargs.get("orientation_range_deg")
                            or tkwargs.get("orientation_deg")
                            or [-10.0, 10.0]
                        )
                        crest_offset = float(tkwargs.get("crest_offset_m", tkwargs.get("crest_offset", 0.0)))
                        sub_terrains = {
                            "flat": HfFurrowsTerrainCfg(
                                proportion=1.0,
                                border_width=0.0,
                                depth_range=(float(depth_rng[0]), float(depth_rng[1])),
                                spacing_range=(float(spacing_rng[0]), float(spacing_rng[1])),
                                orientation_range_deg=(float(orient_rng[0]), float(orient_rng[1])),
                                crest_offset_m=crest_offset,
                            )
                        }
            except Exception as e:
                logger.warning(f"Selected terrain kwargs not applied: {e}")

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            # Allow overriding friction combine behavior from terrain config (default 'multiply').
            _fric_mode = getattr(self.terrain_config, "friction_combine_mode", "multiply")
            _rest_mode = getattr(self.terrain_config, "restitution_combine_mode", "multiply")

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                # Honor the YAML value (Stage-1 uses 0); default to 9 only if absent.
                max_init_terrain_level=getattr(self.terrain_config, "max_init_terrain_level", 9),
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode=_fric_mode,
                    restitution_combine_mode=_rest_mode,
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=getattr(self.terrain_config, "restitution", 0.0),
                    # PhysX compliant normal contact (paper Fn = kn*δ + cn*δ̇).
                    # stiffness>0 enables the spring model; 0 keeps default rigid contact.
                    compliant_contact_stiffness=getattr(self.terrain_config, "compliant_contact_stiffness", 0.0),
                    compliant_contact_damping=getattr(self.terrain_config, "compliant_contact_damping", 0.0),
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:
            _fric_mode = getattr(self.terrain_config, "friction_combine_mode", "multiply")
            _rest_mode = getattr(self.terrain_config, "restitution_combine_mode", "multiply")

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode=_fric_mode,
                    restitution_combine_mode=_rest_mode,
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=getattr(self.terrain_config, "restitution", 0.0),
                    # PhysX compliant normal contact (paper Fn = kn*δ + cn*δ̇).
                    # stiffness>0 enables the spring model; 0 keeps default rigid contact.
                    compliant_contact_stiffness=getattr(self.terrain_config, "compliant_contact_stiffness", 0.0),
                    compliant_contact_damping=getattr(self.terrain_config, "compliant_contact_damping", 0.0),
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
        
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner

        
        
        self.terrain = terrain_config.class_type(terrain_config)
        # Keep terrain.env_origins as (num_rows, num_cols, 3) so the existing
        # `base_task._get_env_origins` indexing (`terrain_origins[levels, types]`)
        # still works and yields (num_envs, 3).
        self.terrain.env_origins = self.terrain.terrain_origins

        # Pack a per-env spawn anchor onto the actual terrain mesh. For
        # generator-type heightfields, terrain.terrain_origins is shaped
        # (rows, cols, 3) and gives sub-terrain centers; we distribute
        # multiple envs per tile via env_spacing-based local offsets and
        # raycast the ground Z so the spawn anchor sits on the mesh. Stored
        # on self._packed_env_origins so `create_envs` can use it without
        # mutating terrain.env_origins (which other code expects in 2D shape).
        self._packed_env_origins = None
        if getattr(self.terrain, "terrain_origins", None) is not None:
            self._packed_env_origins = self._pack_env_origins_in_terrain(self.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])

        # Optional: spawn low-friction patches to emulate wet/loose areas.
        try:
            patch_cfg = getattr(self.terrain_config, "patchy_friction", None)
            if patch_cfg and getattr(patch_cfg, "enabled", False):
                self._spawn_low_friction_patches(patch_cfg)
        except Exception as e:
            # Keep training robust even if the feature is unavailable on this stack
            logger.warning(f"Skipping friction patches: {e}")

        # Optional: decorative maize plants for the agricultural-demo eval. No
        # collision, no physics — purely cosmetic. Plants live under
        # /World/maize_plants/ (NOT under /World/envs/), so they share across
        # all envs and are skipped for performance during high-num_envs training.
        try:
            maize_cfg = getattr(self.terrain_config, "maize_plants", None)
            if maize_cfg and getattr(maize_cfg, "enabled", False):
                self._spawn_maize_plants(maize_cfg)
        except Exception as e:
            logger.warning(f"Skipping maize plants: {e}")

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))

        # ------------------------------------------------------------------
        # DFH (Deformable Furrowed Heightfield) — optional standalone extension.
        # Activated only when terrain.dfh_enabled is set in the Hydra config.
        # Silently skipped if the ext_dfh package is not installed.
        # ------------------------------------------------------------------
        self._dfh = None
        if getattr(self.terrain_config, "dfh_enabled", False):
            try:
                from ext_dfh.integration import build_dfh_adapter
                from omegaconf import OmegaConf
                dfh_dict = OmegaConf.to_container(self.terrain_config.dfh, resolve=True)
                foot_names = list(getattr(self.config.robot, "contact_bodies", []))
                if not foot_names:
                    foot_names = ["left_ankle_link", "right_ankle_link"]
                self._dfh = build_dfh_adapter(
                    num_envs=self.scene.cfg.num_envs,
                    dfh_config_dict=dfh_dict,
                    robot=self._robot,
                    contact_sensor=self.contact_sensor,
                    foot_body_names=foot_names,
                )
                # Sample a per-env furrow orientation matching the parent
                # furrows generator's ``orientation_range_deg``. The generator
                # picks one theta per sub-tile, so per-env (not per-cell)
                # sampling is the right granularity here. Broadcast to the
                # ``(num_envs, grid_h, grid_w)`` per-cell tensor expected by
                # :meth:`DFHTerrainLayer.initialize`.
                furrow_dir = None
                try:
                    kwargs_cfg = getattr(self.terrain_config, "terrain_kwargs", {}) or {}
                    rng = kwargs_cfg.get("orientation_range_deg", None) if isinstance(kwargs_cfg, dict) else getattr(kwargs_cfg, "orientation_range_deg", None)
                    if rng is not None and len(rng) == 2:
                        lo, hi = float(rng[0]), float(rng[1])
                        n_envs = self.scene.cfg.num_envs
                        gh = self._dfh.layer.cfg.grid_h
                        gw = self._dfh.layer.cfg.grid_w
                        per_env_theta = (
                            lo
                            + (hi - lo)
                            * torch.rand(n_envs, device=self._dfh.layer.device)
                        )
                        furrow_dir = per_env_theta.view(n_envs, 1, 1).expand(n_envs, gh, gw).contiguous()
                except Exception as _e:
                    furrow_dir = None
                self._dfh.initialize(furrow_direction_deg=furrow_dir)

                # Force-coupled DFH: per-foot tangential drag is applied each
                # physics step so per-env DFH state actually affects the
                # robot, even when the visual heightfield writeback is off
                # or globally aliased across envs.
                from ext_dfh.integration import build_force_apply_for_robot
                force_apply = build_force_apply_for_robot(
                    robot=self._robot,
                    foot_body_names=foot_names,
                    num_envs=self.scene.cfg.num_envs,
                )
                self._dfh.attach_force_apply(force_apply)
                # Lazily resolve env_origins (set in ``create_envs``).
                self._dfh.attach_env_origins_provider(
                    lambda: getattr(self, "env_origins", None)
                )

                # Path A USD writeback — opt-in via ``terrain.dfh.writeback_enabled``;
                # default OFF because the current implementation aliases all
                # envs onto a single shared mesh via cross-env min reduction.
                wb = None
                if bool(dfh_dict.get("writeback_enabled", False)):
                    from ext_dfh.writeback import build_writeback
                    wb = build_writeback(
                        isaacsim_terrain=self.terrain,
                        terrain_prim_path=terrain_config.prim_path,
                        horizontal_scale_m=self._dfh.layer.cfg.horizontal_scale_m,
                        dfh_grid_h=self._dfh.layer.cfg.grid_h,
                        dfh_grid_w=self._dfh.layer.cfg.grid_w,
                        device=self._dfh.layer.cfg.device,
                    )
                    if wb is not None:
                        self._dfh.attach_physx_writeback(wb)
                logger.info(
                    f"DFH terrain layer attached: feet={foot_names}, "
                    f"grid={self._dfh.layer.cfg.grid_h}x{self._dfh.layer.cfg.grid_w}, "
                    f"force_coupling={'on' if self._dfh.force_coupling_enabled else 'off'}, "
                    f"writeback={'on' if wb is not None else 'off'}"
                )
            except ImportError:
                logger.warning(
                    "terrain.dfh_enabled=True but ext_dfh package not importable; "
                    "running with rigid heightfield."
                )
            except Exception as e:
                logger.warning(f"DFH attach failed: {e}; running with rigid heightfield.")

    # ----------------------------------------------------------------- DFH

    def dfh_reset(self, env_ids: torch.Tensor) -> None:
        """Reset the DFH plastic buffer for the given envs (no-op if DFH off)."""
        if self._dfh is not None:
            self._dfh.on_reset(env_ids)

    def dfh_randomize(self, env_ids: torch.Tensor, ranges) -> None:
        """Sample per-env DFH parameters for the given envs.

        ``ranges`` is the resolved ``domain_rand.dfh_param_ranges`` dict (or
        ``None``). No-op when DFH is disabled or ``ranges`` is empty.
        """
        if self._dfh is None or not ranges:
            return
        try:
            self._dfh.randomize_envs(env_ids, ranges)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"DFH randomize_envs failed: {e}")


    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            from omni.isaac.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1. / self.simulator_config.sim.fps
        
    
    def setup_terrain(self, mesh_type):
        pass

    def _spawn_low_friction_patches(self, patch_cfg):
        """Spawn thin static cuboids with low friction to create patchy zones.
        Requires Omni Isaac Lab's RigidObject/Material APIs at runtime.
        """
        try:
            from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
            import omni.isaac.lab.sim as sim_utils
        except Exception as e:
            raise RuntimeError("RigidObject API not available") from e

        # Derive counts and sizes
        cov_low, cov_high = float(patch_cfg["coverage_fraction"][0]), float(patch_cfg["coverage_fraction"][1])
        size_x_rng = patch_cfg.get("patch_size_m", [1.0, 2.0])
        low_mu = float(patch_cfg.get("low_mu", 0.2))

        # Terrain footprint per env
        L = float(self.terrain_config.terrain_length)
        W = float(self.terrain_config.terrain_width)
        area = L * W
        # Choose total area to cover per env
        coverage = np.random.uniform(cov_low, cov_high)
        target_area = coverage * area

        # Per-env patches under /World/envs/env_*/
        for env_id in range(self.scene.cfg.num_envs):
            placed_area = 0.0
            patch_idx = 0
            while placed_area < target_area and patch_idx < 16:
                sx = float(np.random.uniform(size_x_rng[0], size_x_rng[1]))
                sy = float(np.random.uniform(size_x_rng[0], size_x_rng[1]))
                sz = 0.02  # thin slab

                # random position within env tile (centered at (0,0) in each env frame)
                x = float(np.random.uniform(-0.5 * L + 0.5 * sx, 0.5 * L - 0.5 * sx))
                y = float(np.random.uniform(-0.5 * W + 0.5 * sy, 0.5 * W - 0.5 * sy))
                z = 0.0

                # physics material for the patch
                mat = sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=low_mu,
                    dynamic_friction=low_mu,
                    restitution=0.0,
                )

                spawn = sim_utils.CuboidCfg(
                    size=(sx, sy, sz),
                    physics_material=mat,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Basic/Plastic_PVC.mdl", project_uvw=True
                    ),
                )

                obj_cfg = RigidObjectCfg(
                    # Note: These prim paths are environment-replicated by InteractiveScene
                    prim_path=f"/World/envs/env_{env_id}/friction_patch_{patch_idx}",
                    spawn=spawn,
                )
                obj = RigidObject(obj_cfg)
                # place into scene
                self.scene.rigid_objects[f"patch_{env_id}_{patch_idx}"] = obj
                # initial pose
                obj.set_world_pose(position=np.array([x, y, z]))

                placed_area += sx * sy
                patch_idx += 1

            # Optional single surprise patch
            surprise = patch_cfg.get("surprise_patches", None)
            if surprise and surprise.get("enabled", False):
                sx, sy = [float(v) for v in surprise.get("size_m", [2.0, 2.0])]
                sz = 0.02
                x = float(np.random.uniform(-0.5 * L + 0.5 * sx, 0.5 * L - 0.5 * sx))
                y = float(np.random.uniform(-0.5 * W + 0.5 * sy, 0.5 * W - 0.5 * sy))
                z = 0.0
                low_mu_surprise = float(surprise.get("low_mu", 0.2))
                mat = sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=low_mu_surprise,
                    dynamic_friction=low_mu_surprise,
                    restitution=0.0,
                )
                spawn = sim_utils.CuboidCfg(
                    size=(sx, sy, sz),
                    physics_material=mat,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                )
                obj_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_{env_id}/friction_patch_surprise",
                    spawn=spawn,
                )
                obj = RigidObject(obj_cfg)
                self.scene.rigid_objects[f"patch_surprise_{env_id}"] = obj
                obj.set_world_pose(position=np.array([x, y, z]))


    def _spawn_maize_plants(self, plants_cfg) -> None:
        """Spawn decorative (no collision, no physics) maize plants.

        Plants are imported from `humanoidverse/data/assets/maize/`. Their
        (x, y, yaw) positions come from a virtual_maize_field generated world.
        First call also converts the COLLADA meshes to USD in-place — neither
        `omni.kit.asset_converter` nor USD's stage reference loader handle
        DAE files, so we build the USDs directly via pxr + trimesh while a
        SimulationApp is up (which is when this method runs).

        Config (under `terrain.maize_plants`):
            enabled:   bool (gate)
            max_count: int  (cap, default 200)
            scale:     float (mesh scale, default 0.683 — matches upstream model.sdf)
            z_offset:  float (m, default 0.0)
        """
        import json
        from math import cos, sin
        from pathlib import Path

        import omni.isaac.lab.sim as sim_utils

        assets_dir = Path(__file__).resolve().parents[2] / "data" / "assets" / "maize"
        positions_file = assets_dir / "maize_positions.json"
        if not positions_file.exists():
            logger.warning(f"maize positions JSON missing at {positions_file}; "
                           "run humanoidverse/data/assets/maize/extract_positions.py")
            return

        positions = json.loads(positions_file.read_text())
        max_count = int(plants_cfg.get("max_count", 200))
        positions = positions[:max_count]

        # Ensure USDs exist (one-time DAE→USD via pxr).
        usd_paths = {}
        for model_id in ("maize_01", "maize_02"):
            usd_path = assets_dir / f"{model_id}.usd"
            if not usd_path.exists():
                dae_path = assets_dir / "meshes" / f"{model_id}.dae"
                tex_path = assets_dir / "materials" / "textures" / f"{model_id}.png"
                self._convert_dae_to_usd_pxr(
                    name=model_id,
                    dae_path=dae_path,
                    out_path=usd_path,
                    texture_path=tex_path if tex_path.exists() else None,
                )
            usd_paths[model_id] = str(usd_path)

        scale_val = float(plants_cfg.get("scale", 0.683))
        z_offset = float(plants_cfg.get("z_offset", 0.0))

        # Plants spawn under /World/, not under /World/envs/env_*/, so they do
        # NOT get cloned/replicated per-env. The cached positions are in the
        # virtual_maize_field world frame (centered around origin), but env_0
        # actually lives at the FAR CORNER of the multi-tile terrain grid (eg
        # (-62.5, -62.5)). Offset every plant by env_0's origin so the field
        # surrounds Hunter wherever the cloner placed him.
        try:
            env_origin = self._packed_env_origins[0]
            offset_x = float(env_origin[0])
            offset_y = float(env_origin[1])
        except (AttributeError, IndexError, TypeError):
            offset_x = offset_y = 0.0
        logger.info(f"maize plants anchored at env_0 origin ({offset_x:.2f}, {offset_y:.2f})")

        for i, p in enumerate(positions):
            usd_cfg = sim_utils.UsdFileCfg(
                usd_path=usd_paths[p["model"]],
                scale=(scale_val, scale_val, scale_val),
                # No rigid_props, no collision_props — purely decorative.
            )
            yaw = p["yaw"]
            qw, qz = cos(yaw / 2.0), sin(yaw / 2.0)
            usd_cfg.func(
                prim_path=f"/World/maize_plants/p_{i:04d}",
                cfg=usd_cfg,
                translation=(float(p["x"]) + offset_x, float(p["y"]) + offset_y, z_offset),
                orientation=(qw, 0.0, 0.0, qz),
            )

        logger.info(f"Spawned {len(positions)} maize plants under /World/maize_plants/")


    def _convert_dae_to_usd_pxr(self, name, dae_path, out_path, texture_path) -> None:
        """Author a textured USD stage from a COLLADA mesh using trimesh + pxr.

        Called inside the live SimulationApp (where pxr is on the path). One-off
        per asset — caches results on disk next to the DAE.
        """
        import trimesh
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        if not dae_path.exists():
            logger.warning(f"DAE missing at {dae_path}; cannot convert {name}")
            return

        mesh = trimesh.load(str(dae_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            logger.warning(f"{dae_path} has no faces; skipping {name}")
            return
        logger.info(f"converting {name}: v={len(mesh.vertices)} f={len(mesh.faces)} "
                    f"texture={'ok' if texture_path else 'none'}")

        stage = Usd.Stage.CreateNew(str(out_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        root = UsdGeom.Xform.Define(stage, f"/{name}")
        stage.SetDefaultPrim(root.GetPrim())

        mesh_prim = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
        mesh_prim.CreatePointsAttr().Set([tuple(v) for v in mesh.vertices.astype(float)])
        mesh_prim.CreateFaceVertexIndicesAttr().Set(mesh.faces.astype(int).flatten().tolist())
        mesh_prim.CreateFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Face-varying UVs (if trimesh extracted any).
        visual = getattr(mesh, "visual", None)
        uvs = getattr(visual, "uv", None) if visual is not None else None
        if uvs is not None and len(uvs) == len(mesh.vertices):
            import numpy as np
            primvars = UsdGeom.PrimvarsAPI(mesh_prim.GetPrim())
            st = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            face_uvs = np.asarray(uvs, dtype=float)[mesh.faces.flatten()]
            st.Set([tuple(uv) for uv in face_uvs])

        if texture_path is not None:
            mat = UsdShade.Material.Define(stage, f"/{name}/material")
            pbr = UsdShade.Shader.Define(stage, f"/{name}/material/pbr")
            pbr.CreateIdAttr("UsdPreviewSurface")
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            st_reader = UsdShade.Shader.Define(stage, f"/{name}/material/st_reader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, f"/{name}/material/diffuse_tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )
            mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)

        stage.GetRootLayer().Save()
        logger.info(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


    def load_assets(self):
        '''
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        '''

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")     
        # isaacsim only support matching joint names without "joint" postfix

        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 1.05),
        #     joint_pos={
        #         ".*_hip_yaw": 0.0,
        #         ".*_hip_roll": 0.0,
        #         ".*_hip_pitch": -0.28,  # -16 degrees
        #         ".*_knee": 0.79,  # 45 degrees
        #         ".*_ankle": -0.52,  # -30 degrees
        #         "torso": 0.0,
        #         ".*_shoulder_pitch": 0.28,
        #         ".*_shoulder_roll": 0.0,
        #         ".*_shoulder_yaw": 0.0,
        #         ".*_elbow": 0.52,
        #     },
        #     joint_vel={".*": 0.0},
        # ),

        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        #     ),
        # ),

        # Find the indices of configured joints/bodies inside IsaacSim's internal (BFS) ordering.
        # preserve_order=True returns indices for each name in the order given by the config, i.e.,
        #   dof_ids[cfg_idx] = bfs_idx
        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        # Also capture the BFS-ordered view for clearer diagnostics.
        try:
            bfs_dof_ids, bfs_dof_names = self._robot.find_joints(dof_names_list, preserve_order=False)
        except Exception:
            bfs_dof_ids, bfs_dof_names = None, None


        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, humanoidverse config)
            # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

    
        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        '''
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19], ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        '''
        
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # Build remapping helpers: cfg_idx -> bfs_idx and bfs_idx -> cfg_idx
        # These ensure all read/write paths stay consistent even when orders differ.
        self.cfg_to_bfs = torch.as_tensor(self.dof_ids, device=self.sim_device)
        try:
            self.bfs_to_cfg = torch.argsort(self.cfg_to_bfs)
        except Exception:
            # Fallback on CPU if needed (e.g., before devices are fully initialized)
            self.bfs_to_cfg = torch.argsort(self.cfg_to_bfs.cpu()).to(self.sim_device)

        # Warning (with actionable details) if config order != IsaacSim BFS order
        if self.dof_ids != list(range(self.num_dof)):
            msg_lines = [
                "The order of robot.dof_names (config) does not match IsaacSim's DOF order.",
                "Actions/PD/obs are remapped safely by name, but reordering YAML is recommended for clarity.",
                f"cfg->bfs index map: {self.dof_ids}",
            ]
            if bfs_dof_names is not None:
                msg_lines.append(f"Config DOF order: {self.robot_config.dof_names}")
                msg_lines.append(f"IsaacSim DOF order: {bfs_dof_names}")
                msg_lines.append("Tip: reorder dof_names and associated limit lists in"
                                  " humanoidverse/config/robot/hunter/hunter.yaml to match IsaacSim order above.")
            logger.warning("\n".join(msg_lines))
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"
       
        
        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names
        

    def _pack_env_origins_in_terrain(self, terrain) -> torch.Tensor:
        """Distribute num_envs spawn anchors over the available sub-terrains.

        Each tile gets up to ``ceil(num_envs / num_tiles)`` envs laid out on a
        local grid of ``env_spacing``. Z is set from a downward raycast onto
        the terrain's warp mesh when available so the spawn point sits on the
        actual ground surface.
        """
        base = terrain.terrain_origins.reshape(-1, 3).to(self.sim_device)
        num_envs = self.scene.cfg.num_envs
        num_tiles = base.shape[0]
        per_tile = math.ceil(num_envs / num_tiles)

        spacing = float(self.scene.cfg.env_spacing)
        length = float(self.terrain_config.terrain_length)
        width = float(self.terrain_config.terrain_width)

        nx = max(1, int(math.floor(length / spacing)))
        ny = max(1, int(math.floor(width / spacing)))
        if nx * ny < per_tile:
            raise ValueError(
                f"Cannot place {num_envs} envs on {num_tiles} tiles with "
                f"tile={length}x{width} and env_spacing={spacing}: "
                f"capacity={num_tiles * nx * ny}."
            )

        xs = (torch.arange(nx, device=self.sim_device, dtype=torch.float32)
              - (nx - 1) / 2.0) * spacing
        ys = (torch.arange(ny, device=self.sim_device, dtype=torch.float32)
              - (ny - 1) / 2.0) * spacing
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        local = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)[:per_tile]

        env_ids = torch.arange(num_envs, device=self.sim_device)
        tile_ids = env_ids % num_tiles
        slot_ids = torch.div(env_ids, num_tiles, rounding_mode="floor")

        env_origins = base[tile_ids].clone()
        env_origins[:, :2] += local[slot_ids]

        # Snap Z to the actual mesh surface via warp raycast when available.
        # Use a FOOTPRINT of rays (not a single anchor point): on a corrugated
        # furrow surface a single-point raycast can land in a trough/crest that
        # the robot's spread feet do not stand on, leaving the feet unsupported
        # (0 N foot contact, robot topples). Snapping Z to the MAX hit over the
        # stance footprint guarantees the whole footprint is at-or-below spawn,
        # so the feet settle cleanly onto the surface. A small clearance lets
        # the robot drop the last few mm into clean contact.
        try:
            from omni.isaac.lab.utils.warp import raycast_mesh
            warp_meshes = getattr(terrain, "warp_meshes", None) or {}
            mesh = warp_meshes.get("terrain") or next(iter(warp_meshes.values()), None)
            if mesh is not None:
                # 3x3 footprint, r = 0.25 m: covers Hunter's stance without
                # reaching into the neighbouring furrow (spacing ~2.7-3.2 m).
                r = 0.25
                offs = torch.tensor(
                    [[0.0, 0.0], [-r, -r], [-r, 0.0], [-r, r],
                     [0.0, -r], [0.0, r], [r, -r], [r, 0.0], [r, r]],
                    device=self.sim_device, dtype=torch.float32,
                )
                n_off = offs.shape[0]
                starts = env_origins[:, None, :].expand(-1, n_off, -1).clone()
                starts[:, :, :2] += offs[None, :, :]
                starts[:, :, 2] = 100.0
                dirs = torch.zeros_like(starts)
                dirs[:, :, 2] = -1.0
                flat_s = starts.reshape(-1, 3)
                flat_d = dirs.reshape(-1, 3)
                hits = raycast_mesh(flat_s.unsqueeze(0), flat_d.unsqueeze(0), mesh)[0][0]
                hits = hits.to(self.sim_device).reshape(-1, n_off, 3)
                hit_z = hits[:, :, 2]
                valid = torch.isfinite(hit_z)
                patch_z = torch.where(valid, hit_z, torch.full_like(hit_z, -1e9)).amax(dim=1)
                has_hit = patch_z > -1e8
                spawn_clearance = 0.03  # m — settle the last few mm into contact
                env_origins[has_hit, 2] = patch_z[has_hit] + spawn_clearance
        except Exception as _e:
            logger.warning(f"[pack env_origins] raycast snap-to-mesh skipped: {_e}")

        logger.info(
            f"[pack env_origins] envs={num_envs} tiles={num_tiles} per_tile={per_tile} "
            f"xy_range=[{env_origins[:, :2].min().item():.2f}, "
            f"{env_origins[:, :2].max().item():.2f}] "
            f"z_range=[{env_origins[:, 2].min().item():.4f}, "
            f"{env_origins[:, 2].max().item():.4f}]"
        )
        return env_origins

    def create_envs(self, num_envs, env_origins, base_init_state):

        self.num_envs = num_envs
        # Prefer the packed (num_envs, 3) tensor we built in _setup_scene that
        # snaps each env to a sub-terrain center plus a local env_spacing
        # slot. Fall back to scene.env_origins (uniform cloner grid) for
        # plane terrains where there are no per-tile origins.
        packed = getattr(self, "_packed_env_origins", None)
        if (
            torch.is_tensor(packed)
            and packed.dim() == 2
            and packed.shape == (num_envs, 3)
        ):
            self.env_origins = packed.to(self.sim_device)
            logger.info(
                f"[create_envs] using packed env_origins shape={tuple(packed.shape)}"
            )
        else:
            scene_origins = self.scene.env_origins
            self.env_origins = scene_origins.to(self.sim_device)
            logger.info(
                f"[create_envs] using scene.env_origins shape={tuple(scene_origins.shape)}"
            )
        self.base_init_state = base_init_state

        # Always retarget the viewport camera at env_0's actual spawn so the
        # user can see the robot when not headless. The default world-origin
        # camera misses the spawn area whenever env_origins are packed onto a
        # real heightfield tile (tens of meters away from origin). Costs
        # nothing in headless training.
        try:
            eo = self.env_origins[0].detach().cpu().tolist()
            target = [float(eo[0]), float(eo[1]), float(eo[2]) + 0.7]
            cam = [target[0] + 3.0, target[1] + 3.0, target[2] + 1.8]
            self.sim.set_camera_view(cam, target)
            logger.info(
                f"[viewport] camera set to look at env_0 origin "
                f"target={target} cam={cam}"
            )
        except Exception as _e:
            logger.warning(f"[viewport] camera retarget skipped: {_e}")

        return self.scene, self._robot
    
    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices
                
    def prepare_sim(self):
        self.refresh_sim_tensors() # initialize tensors

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        ############################################################################################
        # TODO: currently, we only consider the robot root state, ignore other objects's root states
        ############################################################################################
        self.all_root_states = self._robot.data.root_state_w  # (num_envs, 13)
        
        self.robot_root_states = self.all_root_states # (num_envs, 13)
        self.base_quat = self.robot_root_states[:, [4, 5, 6, 3]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        
        # Map raw BFS-ordered joint_state into config order using cfg->bfs mapping
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids] # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        self.contact_forces = self.contact_sensor.data.net_forces_w # (num_envs, num_bodies, 3)

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    def apply_torques_at_dof(self, torques):
        # Write torques provided in config order to the correct BFS joints in the simulator.
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        self._robot.write_root_pose_to_sim(root_states[set_env_ids, :7], set_env_ids)
        self._robot.write_root_velocity_to_sim(root_states[set_env_ids, 7:], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)
    
    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # DFH per-step update: read foot contacts, advance the plastic buffer,
        # (v0.3) push deformation back into PhysX.  Runs BEFORE sim.step so that
        # any heightfield write-back takes effect on the upcoming integration.
        if self._dfh is not None:
            try:
                self._dfh.on_physics_step()
            except Exception as e:
                logger.warning(f"DFH step skipped: {e}")

        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim 
        self.scene.update(dt=1./self.simulator_config.sim.fps)
    
    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller


    def render(self, sync_frame_time=True):
        pass

     # debug visualization
    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [(   start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)
