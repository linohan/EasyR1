rollout_batchsize：一个batch内有多少个rollout轨迹（batch的样本数*每条样本的采样数）
global_batchsize：一个batch的样本数



rollout_batchsize//global_batchsize是一个data batch内的policy更新次数