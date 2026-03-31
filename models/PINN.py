from models.UNet import *


class PINNConfig(UNetConfig):
    """"""
    def __init__(
        self,
        C_material,
        C_dynamics,
        base_channel,
        eps,
        **kwargs
    ):
        """"""
        super().__init__(base_channel, eps)
        self.C_material = C_material
        self.C_dynamics = C_dynamics


class PINN(UNet):
    """"""
    def __init__(self, config:PINNConfig):
        super(PINN, self).__init__(config)
        self.C_material = config.C_material
        self.C_dynamics = config.C_dynamics

        # Sobel算子
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3))

    def compute_gradient(self, batch):
        """"""
        gradients = []
        for image in batch:
            grad_x = F.conv2d(image.unsqueeze(0), self.sobel_x, padding=1)
            grad_y = F.conv2d(image.unsqueeze(0), self.sobel_y, padding=1)
            grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            gradients.append(grad_magnitude.mean())
        return torch.mean(torch.Tensor(gradients))

    def forward(self, x, r, y):
        output = super().forward(x, r, y)
        loss, pred = output.loss, output.logits

        loss += self.C_material * (torch.mean(pred) - torch.mean(r[:, 1]) * 0.03)**2 # 0.03为根据实际估测的比例值
        loss += self.C_dynamics * (self.compute_gradient(pred) - self.compute_gradient(y))**2 # 对梯度进行惩罚，要求预测的水量尽量连续分布

        return SequenceClassifierOutput(loss=loss, logits=pred)

