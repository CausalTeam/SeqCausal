# SeqCausal

SeqCausal 是一个用于序贯因果推断的工具包，专注于简单场景下的应用。序贯因果推断在开始时使用少量特征进行推断，如果无法获得足够准确的结果，再逐步引入更多特征，以此节约计算成本。
# 场景简介

序贯因果推断与传统因果推断的主要区别在于特征使用的灵活性。传统因果推断方法要求固定数目的特征，而序贯因果推断则允许对不同的样本使用不同数量的特征，从而在保持准确度的同时节省计算资源。
# 如何开始
## 安装环境

请运行以下命令以安装必要的依赖：

bash

pip install -r requirements.txt

## 运行实验

请运行以下命令以开始实验：

bash

python main.py

## 更改参数

如果需要更改参数，可以编辑 main.py 文件中第28行到第65行的代码。通过调整这些参数，可以选择不同的数据集和模型以优化超参数。
# 代码结构

项目代码分为以下几个主要部分：
## Dataset

dataset 文件夹中实现了多个模拟数据集的设置，并包括了 IHDP 和 ACIC2016 数据集。其主要功能是：

    接收 args 参数
    输出分割好的训练集、验证集和测试集

## Model

model 文件夹定义了模型的具体结构，目前实现了 MLP 网络和 JAFA 网络。其主要功能包括：

    输入特征 X，输出决策价值函数 q
    输入特征 X，输出预期输出 y1 和 y0

## Inference

inference 文件定义了推断的具体方法，并包含训练和测试推断器的步骤。
## Agent

agent 文件定义了决策的具体方法，并包含训练和测试决策器的步骤。
## Environment

environment 文件定义了给予奖励的方式。
# 贡献指南

如果你有兴趣为 SeqCausal 做出贡献，请遵循以下步骤：

    Fork 本仓库
    创建你的功能分支 (git checkout -b feature/AmazingFeature)
    提交你的修改 (git commit -m 'Add some AmazingFeature')
    推送到分支 (git push origin feature/AmazingFeature)
    提交一个 Pull Request

# 许可证

该项目使用 MIT 许可证。详情请参见 LICENSE 文件。
作者

    [tianyuancunyan]

