import numpy as np
import mujoco.viewer
import gym
from gym import spaces

# 建立ur5e机械臂强化学习环境
# 任务：从定点出发使末端执行器到达某一范围内的目标（GoRL）
# 初始状态：[-1.57, -1.34, 2.65, -1.3, 1.55, 0]
# 对应机械臂初始位姿[-0.14, 0.3, 0.1, 3.14, 0, 1.57]
# 目标位置：ee_target_pos = [-0.1 + np.random.uniform(-0.05, 0.05), 0.6 + np.random.uniform(-0.05, 0.05),
#                        0.15 + np.random.uniform(-0.05, 0.05)]
# 目标姿态：ee_target_euler = [3.14, 0, 1.57]
# 动作空间：6个关节电机控制角度增量，范围-1到1，与当前实际角度相加后得到可用的关节电机控制量
# 关节角度限幅：shoulder_pan:-pi~pi    shoulder_lift:-pi~pi     elbow:-pi~pi
#               wrist_1/wrist_2/wrist_3:-pi~pi
# 状态空间：12个状态，末端执行器位置+实际关节角+目标位置
# action采用角度制，其余均采用弧度制

class UR5E_Env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
        self.data = mujoco.MjData(self.model)
        self.start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
        self.ee_id = self.model.site("attachment_site").id  # 末端执行器标号

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.iteration = 0  # 执行步数
        self.distance = 0
        self.distance_last = 0
        self.on_goal = False
        self.on_goal_count = 0
        self.done = False

    def viewer_init(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = [0, 0.5, 0.5]
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -30

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.start_joints  # 确保渲染一开始机械臂便处于起始位置，而非MJCF中的默认位置
        self.data.ctrl[:6] = self.start_joints  # 初始控制量与实际角度保持一致
        mujoco.mj_step(self.model, self.data)

        self.ee_target_pos = [0. + np.random.uniform(-0.05, 0.05),
                              0.5 + np.random.uniform(-0.05, 0.05),
                              0.15 + np.random.uniform(-0.05, 0.05)]  # 重置目标点
        self.model.geom('target_point').pos = self.ee_target_pos

        self.iteration = 0  # 执行步数
        self.on_goal = False
        self.on_goal_count = 0
        self.done = False

        # 状态空间：12个状态，关节角+目标位置+目标位置与末端位置差值+是否接触目标
        obs = np.hstack((self.data.qpos[:6], self.ee_target_pos,
                            self.data.site_xpos[self.ee_id] - self.ee_target_pos, 0.),
                            dtype=np.float32).flatten()
        return obs

    def step(self, action):
        reward = 0
        self.data.qpos = np.clip(self.data.qpos + action[:6] * 0.5 / 180 * np.pi, -np.pi, np.pi)
        self.data.ctrl[:6] = self.data.qpos
        # 确保step后关节实际角度接近电机设定角度，如果有碰撞则提前结束
        mujoco.mj_step(self.model, self.data)
        # 计算距离奖励
        reward = -np.linalg.norm(self.data.site_xpos[self.ee_id] - self.ee_target_pos)
        reward -= 0.3 * self.data.ncon
        # 计算完成奖励：发生碰撞或者超过最大步数，完成奖励-1；末端到达预定位置，完成奖励+10
        if self.iteration >= 500:
            self.done = True
        elif np.allclose(self.data.site_xpos[self.ee_id], self.ee_target_pos, atol=0.02):
            reward += 10
            self.on_goal = True
            self.on_goal_count += 1
            if self.on_goal_count > 50:
                self.on_goal_count = 0
                self.done = True

        obs = np.hstack((self.data.qpos[:6], self.ee_target_pos,
                         self.data.site_xpos[self.ee_id] - self.ee_target_pos,
                         [1. if self.on_goal else 0.]), dtype=np.float32).flatten()
        self.iteration += 1
        self.on_goal = False
        return obs, reward, self.done

    def render(self):
        self.viewer.sync()



