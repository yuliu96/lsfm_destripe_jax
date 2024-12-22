import torch


class update_torch:
    def __init__(
        self,
        network,
        Loss,
        learning_rate,
    ):
        self.learning_rate = learning_rate
        self.loss = Loss
        self._network = network

    def opt_init(self, net_params):
        return torch.optim.Adam(net_params, lr=self.learning_rate)

    def __call__(
        self,
        step,
        params,
        optimizer,
        aver,
        xf,
        y,
        mask_dict,
        hy,
        targets_f,
        targetd_bilinear,
    ):
        optimizer.zero_grad()
        l, A = self.loss(
            self._network,
            {
                "aver": aver,
                "Xf": xf,
                "target": y,
                "target_hr": hy,
                "coor": mask_dict["coor"],
            },
            targetd_bilinear,
            mask_dict,
            hy,
            targets_f,
        )
        l.backward()
        optimizer.step()
        return l, self._network.parameters(), optimizer, A
