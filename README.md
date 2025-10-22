# TD3-Reinforcement-Learning-on-UR5E
在Mujoco环境使用TD3强化学习控制UR5E机械臂操作

说明：

./model/:存放UR5E机械臂模型，来自Mujoco官方仓库

./ur5e.xml:UR5E机械臂xml文件

./UR5E_rl.py:定义强化学习环境，包含状态空间奖励函数等

./train.py:TD3算法实现并训练模型

./test.py:测试模型运行效果，显示可视化界面

./TD3_UR5E_Actor.pth:训练2000轮后的模型参数

./TD3_UR5E.png:训练第1001-2000轮时每一轮获得的奖励

模型在训练2000轮后可稳定收敛

使用到的python库：

matplotlib

numpy

mujoco

torch

tqdm

gym
