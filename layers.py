import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from itertools import combinations


class SConv2d_v0(nn.Module):
    # first feasible version of SConv2d(2D convolution using MAJ to replace addition)
    # it is slow in 2 ways:
    # (1) implementation of MAJ/stackMAJ is primitive, not vectorized (batch-processing support)
    # (2) implementation of 2D convolution uses too much for-loop
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)

        kh, kw = self.kernel_size
        weight_shape = (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.maj_dim = kernel_size

    @staticmethod
    def MAJ(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        result = 0
        majority_threshold = (n + 1) // 2

        for k in range(majority_threshold, n + 1):
            for combo in combinations(range(n), k):
                term = 1
                for i in range(n):
                    if i in combo:
                        term *= (1 + x[i])
                    else:
                        term *= (1 - x[i])
                result += term

        return result / (2 ** (n - 1)) - 1

    @staticmethod
    def stackMAJ(x: torch.Tensor, maj_dim: int) -> torch.Tensor:
        while x.size(0) >= maj_dim:
            chunks = torch.chunk(x, int(x.size(0) / maj_dim))
            x = torch.tensor([SConv2d_v0.MAJ(chunk) for chunk in chunks])
        return x

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # add padding
        if ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph))

        # output size
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        # initialize output tensor
        output = torch.zeros(N, self.out_channels, H_out, W_out,
                             dtype=x.dtype, device=x.device)

        # conduct convolution for every output channel
        for out_ch in range(self.out_channels):
            for i in range(H_out):
                for j in range(W_out):

                    # position of current patch
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw

                    patch = x[:, :, h_start:h_end, w_start:w_end]  # [N, C, kh, kw]
                    kernel = self.weight[out_ch]  # [C, kh, kw]

                    # batch processing
                    for batch_idx in range(N):

                        # elementwise multiplication
                        products = patch[batch_idx] * kernel  # [C, kh, kw]

                        # flatten the result from multiplication
                        products_flat = products.flatten()  # [C*kh*kw]

                        # conduct addition with stackMAJ
                        if products_flat.size(0) > 0:
                            if products_flat.size(0) == 1:
                                result = products_flat[0]
                            else:
                                result = self.stackMAJ(products_flat, self.maj_dim)

                                # in case stackMAJ gives more than one output
                                # while result.numel() > 1:
                                #     result = self.stackMAJ(result, self.maj_dim)

                                # result = result.item() if result.numel() == 1 else result.sum()
                                result = result.item()
                        else:
                            result = 0

                        output[batch_idx, out_ch, i, j] = result

        # bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output


class SConv2d_v1(nn.Module):
    # second feasible version of SConv2d
    # no longer use for-loop to compute convolution, use large-scale matrix multiplication instead
    # yet MAJ and stackMAJ not vectorized, thus still slow
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)

        kh, kw = self.kernel_size
        weight_shape = (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.maj_dim = kernel_size

    @staticmethod
    def MAJ(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        result = 0
        majority_threshold = (n + 1) // 2

        for k in range(majority_threshold, n + 1):
            for combo in combinations(range(n), k):
                term = 1
                for i in range(n):
                    if i in combo:
                        term *= (1 + x[i])
                    else:
                        term *= (1 - x[i])
                result += term

        return result / (2 ** (n - 1)) - 1

    @staticmethod
    def stackMAJ(x: torch.Tensor, maj_dim: int) -> torch.Tensor:
        while x.size(0) >= maj_dim:
            chunks = torch.chunk(x, int(x.size(0) / maj_dim))
            x = torch.tensor([SConv2d_v1.MAJ(chunk) for chunk in chunks])
        return x

    @staticmethod
    def batch_stackMAJ(tensor: torch.Tensor, maj_dim: int) -> torch.Tensor:

        N, out_channels, _, L = tensor.size()

        # move the third dimension to the last
        tensor_moved = tensor.movedim(2, -1)
        original_shape = tensor_moved.shape

        # flatten all other dimensions
        flat_tensor = tensor_moved.reshape(-1, original_shape[-1])

        # conduct stackMAJ
        results = []
        for i in range(flat_tensor.size(0)):
            maj_result = SConv2d_v1.stackMAJ(flat_tensor[i], maj_dim)
            assert maj_result.numel() == 1, "result of stackMAJ must be a single number"
            results.append(maj_result.item())

        result = torch.tensor(results, dtype=tensor.dtype, device=tensor.device)
        return result.reshape(N, out_channels, L)

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # output size
        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        # unfold the input tensor using F.unfold, getting tensor [N, C_in*kh*kw, L]
        # the "L" is the number of all possible sliding-window positions
        # the "C_in*kh*kw" is number of pixels of a certain [C_in, kh, kw] region of input tensor
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           padding=self.padding, stride=self.stride)  # [N, C_in*kh*kw, L], where L = H_out * W_out
        weight_flat = self.weight.view(self.out_channels, -1)  # [C_out, C_in*kh*kw]

        '''
        # If we are tring to implement Conv2d manually, what we do next is to do matrix multiplication between weight_falt and pathches:
        out_unfolded = weight_flat @ patches    # [N, C_out, L]
        # Tip: here, a certain element out_unfolded[n, m, k] means the value of k-th position convolution in the m-th out channel of the n-th sample 
        
        # and then plus the bias:
        if self.bias is not None:
            out_unfolded += self.bias.view(1, -1, 1)
        # finally, reshape the result into [N, C_out, H_out, W_out]:
        out = out_unfolded.view(N, self.out_channels, H_out, W_out)
        return out
        
        # We've successfully implement Conv2d manually thus far.
        '''

        # Here the purpose is to replace the usual "plus" with stackMAJ in 2D convolution,
        # so it's not appropriate to directly muptiply the two tensors. Instead, we do elementwise product first:
        prod = patches.unsqueeze(1) * weight_flat.unsqueeze(0).unsqueeze(-1)  # [N, C_out, C_in*kh*kw, L]
        #      [N, 1, C_in*kh*kw, L]   *       [1, C_out, C_in*kh*kw, 1]
        # a certain element prod[n, m, k, l] means the k-th elementwise product in the l-th sliding-window position in the m-th channel of the n-th sample

        # we shall next conduct stackMAJ on the third dimension of this prod
        out_unfolded = SConv2d_v1.batch_stackMAJ(prod, self.maj_dim)
        out = out_unfolded.view(N, self.out_channels, H_out, W_out)
        return out


class SConv2d(nn.Module):
    # third feasible implementation of SConv2d
    # MAJ/stackMAJ is now vectorized and support batch processing
    # conduct 2D convolution using large scale matrix multiplication, as in SConv2d_v1
    # acceptable speed
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)

        kh, kw = self.kernel_size
        weight_shape = (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.maj_dim = kernel_size

    @staticmethod
    def MAJ(x: torch.Tensor) -> torch.Tensor:
        '''
        vectorized MAJ, support batch processing
        :param x: [batch_size, n]
        :return: result: [batch_size,]
        '''
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, n = x.shape
        majority_threshold = (n + 1) // 2

        # pre-compute 1 + x and 1 - x
        one_plus_x = 1 + x  # shape: (batch_size, n)
        one_minus_x = 1 - x  # shape: (batch_size, n)

        result = torch.zeros(batch_size, dtype=x.dtype, device=x.device)

        # compute for every possible majority number in [majority_threshold, n]
        for k in range(majority_threshold, n + 1):
            # compute for every possible combinations that sums up to k
            for combo in combinations(range(n), k):
                combo_mask = torch.zeros(n, dtype=torch.bool, device=x.device)
                combo_mask[list(combo)] = True

                term = torch.ones(batch_size, dtype=x.dtype, device=x.device)

                # vectorized product
                term *= torch.prod(torch.where(combo_mask, one_plus_x, one_minus_x), dim=1)
                result += term

        return result / (2 ** (n - 1)) - 1

    @staticmethod
    def stackMAJ(x: torch.Tensor, maj_dim: int) -> torch.Tensor:
        '''
        vectorized stackMAJ, support batch processing
        :param x: [batch_size, n]
        :param maj_dim: integer, number of inputs of a single MAJ
        :return: result: [batch_size, 1]
        '''

        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        while x.size(1) >= maj_dim:
            current_length = x.size(1)
            num_chunks = current_length // maj_dim

            if num_chunks == 0:
                break

            # reshape to fit the input shape of MAJ
            reshaped = x[:, :num_chunks * maj_dim].reshape(batch_size * num_chunks, maj_dim)

            # conduct MAJ
            maj_results = SConv2d.MAJ(reshaped)

            # reshape back
            x = maj_results.reshape(batch_size, num_chunks)

        return x

    @staticmethod
    def stackMAJ_conv2d(prod: torch.Tensor, maj_dim: int) -> torch.Tensor:
        '''
        stackMAJ designed for conv2d, support batch processing
        :param prod: [N,C_out, C_in*kh*kw, L]   (conduct stackMAJ to its 3rd dimension)
        :param maj_dim: integer, number of inputs of a single MAJ
        :return: result: [N, C_out, L]
        '''

        N, out_channels, _, L = prod.size()

        # move the target dimension(3rd) to the last, and reshape to fit the input shape of stackMAJ
        prod = prod.movedim(2, -1)  # (N, out_channels, L, C_in*kh*kw)
        flat_tensor = prod.reshape(-1, prod.shape[-1])

        # conduct stackMAJ
        maj_results = SConv2d.stackMAJ(flat_tensor, maj_dim)

        # check size of the maj_result (should be [N*C_out*L, 1]), and turn the column vector into a row vector
        assert maj_results.size(1) == 1, "result of stackMAJ must be a single value"
        maj_results = maj_results.squeeze(1)

        # again reshape back to [N, C_out, L]
        result = maj_results.reshape(N, out_channels, L)

        return result   # [N, C_out, L]

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # output size
        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        # unfold the input tensor using F.unfold, getting tensor [N, C_in*kh*kw, L]
        # the "L" is the number of all possible sliding-window positions
        # the "C_in*kh*kw" is number of pixels of a certain [C_in, kh, kw] region of input tensor
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           padding=self.padding, stride=self.stride, dilation=self.dilation)  # [N, C_in*kh*kw, L], where L = H_out * W_out
        weight_flat = self.weight.view(self.out_channels, -1)  # [C_out, C_in*kh*kw]

        '''
        # If we are tring to implement Conv2d manually, what we do next is to do matrix multiplication between weight_falt and pathches:
        out_unfolded = weight_flat @ patches    # [N, C_out, L]
        # Tip: here, a certain element out_unfolded[n, m, k] means the value of k-th position convolution in the m-th out channel of the n-th sample 

        # and then plus the bias:
        if self.bias is not None:
            out_unfolded += self.bias.view(1, -1, 1)
        # finally, reshape the result into [N, C_out, H_out, W_out]:
        out = out_unfolded.view(N, self.out_channels, H_out, W_out)
        return out

        # We've successfully implement Conv2d manually thus far.
        '''

        # Here the purpose is to replace the usual "plus" with stackMAJ in 2D convolution,
        # so it's not appropriate to directly muptiply the two tensors. Instead, we do elementwise product first:
        prod = patches.unsqueeze(1) * weight_flat.unsqueeze(0).unsqueeze(-1)  # [N, C_out, C_in*kh*kw, L]
        #      [N, 1, C_in*kh*kw, L]   *       [1, C_out, C_in*kh*kw, 1]
        # a certain element prod[n, m, k, l] means the k-th elementwise product in the l-th sliding-window position in the m-th channel of the n-th sample

        # we shall next conduct stackMAJ to its 3rd dimension and squeeze this dimension
        out_unfolded = SConv2d.stackMAJ_conv2d(prod, self.maj_dim)
        out = out_unfolded.view(N, self.out_channels, H_out, W_out)
        return out


class SAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride or kernel_size)  # if stride is None, self.stride=kernel_size
        self.padding = nn.modules.utils._pair(padding)

        self.maj_dim = kernel_size

    @staticmethod
    def stackMAJ_avgpool(patches: torch.Tensor, maj_dim: int) -> torch.Tensor:
        '''
        stackMAJ designed for avgpool2d, support batch processing
        :param patches: [N, C_in, kh*kw, L]
        :param maj_dim: integer, number of inputs of a single MAJ
        :return: result: [N, C_in, L]
        '''
        N, C, _, L = patches.size()
        patches = patches.permute(0, 1, 3, 2)   # [N, C_in, L, kh*kw]
        patches = patches.reshape(-1, patches.shape[-1])    # [N*C_in*L, kh*kw]

        result = SConv2d.stackMAJ(patches, maj_dim) # [N*C_in*L, 1]
        assert result.size(1) == 1, "result of stackMAJ_avgpool must be a single value"
        result = result.reshape(N, C, L)    # [N, C_in, L]
        return result

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # output size
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        # unfold into patches
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           padding=self.padding, stride=self.stride)  # [N, C_in*kh*kw, L], where L = H_out * W_out

        # reshape
        patches = patches.view(N, C, kh * kw, -1)  # [N, C_in, kh*kw, L]
        # if we are to implement AvgPool2d manually, we just need to compute mean value to its 3rd dimension and reshape back:
        # out = patches.mean(dim=2)
        # out = out.view(B, C, H_out, W_out)

        # but here we need to replace "mean" with stackMAJ
        out = SAvgPool2d.stackMAJ_avgpool(patches, maj_dim=self.maj_dim)    # [N, C_in, L]
        out = out.view(N, C, H_out, W_out)
        return out


class SActv(nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, x):
        for _ in range(self.repeats):
            x = -0.5 * x ** 3 + 1.5 * x
        return x


class SLinear(nn.Module):
    def __init__(self, in_features, out_features, maj_dim=3, bias=True):
        # make sure in_features is some power of maj_dim
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self.maj_dim = maj_dim

    @staticmethod
    def stackMAJ_linear(prod: torch.Tensor, maj_dim: int) -> torch.Tensor:
        '''
        stackMAJ designed for linear layer, support batch processing
        :param prod: [batch_size, out_features, in_features]
        :param maj_dim: integer, number of inputs of a single MAJ
        :return: [batch_size, out_features]
        '''
        batch_size, out_features, _ = prod.size()
        prod = prod.reshape(-1, prod.shape[-1])  # [batch_size * out_features, in_features]

        result = SConv2d.stackMAJ(prod, maj_dim)
        assert result.size(1) == 1, "result of stackMAJ must be a single value"
        result = result.squeeze(1)

        result = result.reshape(batch_size, out_features)
        return result

    def forward(self, x):
        '''
        :param x: [batch_size, in_features]
        :return: [batch_size, out_features]
        '''

        # The manual implementation of the standard linear layer is to merely use F.linear(x, self.weight, self.bias)
        # yet the purpose here is to replace addition in matmul with stackMAJ
        # So we need to extract the intermediate quantity of the matrix multiplication,
        # where elementwise product has finished and addition has not.
        prod = x.unsqueeze(1) * self.weight.unsqueeze(0)  # [batch_size, out_features, in_features]
        # here, a certain element prod[n, m, k] means the k-th element pair product of the m-th out_feature of the n-th sample

        # the next task is to conduct stackMAJ to its 3rd dimension and squeeze this dimension.
        out = SLinear.stackMAJ_linear(prod, self.maj_dim)  # [batch_size, out_features]

        if self.bias is not None:
            out = out + self.bias

        return out


if __name__ == '__main__':
    l = SAvgPool2d(kernel_size=3, stride=1, padding=1)
    x = torch.randn(32, 1, 28, 28)
    print(l(x).shape)
