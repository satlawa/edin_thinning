# https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/4


from torch import nn

class Dice_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_Loss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, output, target, eps=1e-7, ignore_index=None, weights=None):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
        output = self.softmax(output)
        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        # numerator
        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        # denominator
        denominator = output + encoded_target
        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        # dice loss
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class IoU_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_Loss, self).__init__()

    def forward(self, output, target, eps=1e-7, ignore_index=None, weights=None):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        # numerator
        intersection = output * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1)
        # denominator
        cardinality = output + encoded_target
        denominator = cardinality - intersection
        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        # IoU loss
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)
