import torch
import torch.nn as nn

class DEQFixedPointFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DEQFixedPointFunc, self).__init__()
        self.linear_layer = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, z, x):
        combined_input = torch.cat((z, x), dim=1)
        out = self.linear_layer(combined_input)
        out = self.activation(out)
        return out

class FixedPointSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, z0, x, max_iter, tol):
        with torch.no_grad():
            z = z0
            for i in range(max_iter):
                z_next = func(z, x)
                if torch.norm(z_next - z, p=2) < tol:
                    z = z_next
                    break
                z = z_next
            else:
                print(f"Warning: Fixed-point solver did not converge after {max_iter} iterations. Norm: {torch.norm(z_next - z, p=2):.4f}")

        ctx.save_for_backward(z, x)
        ctx.func = func
        return z

    @staticmethod
    def backward(ctx, grad_z_out):
        z, x = ctx.saved_tensors
        func = ctx.func

        adjoint_max_iter = 50
        adjoint_tol = 1e-5

        v = grad_z_out.clone()

        for _ in range(adjoint_max_iter):
            with torch.enable_grad():
                z.requires_grad_(True)
                x.requires_grad_(True)
                output_of_func = func(z, x)

            grad_func_z_v = torch.autograd.grad(output_of_func, z, v, retain_graph=True)[0]

            v_next = grad_z_out + grad_func_z_v

            if torch.norm(v_next - v, p=2) < adjoint_tol:
                v = v_next
                break
            v = v_next
        else:
             print(f"Warning: Adjoint fixed-point solver did not converge after {adjoint_max_iter} iterations. Norm: {torch.norm(v_next - v, p=2):.4f}")

        grad_x = torch.autograd.grad(output_of_func, x, v)[0]
        grad_func_params = torch.autograd.grad(output_of_func, func.parameters(), v)

        return (None, None, grad_x, None, None) + grad_func_params

class DEQLayer(nn.Module):
    def __init__(self, func: nn.Module, max_iter: int = 50, tol: float = 1e-4):
        super(DEQLayer, self).__init__()
        self.func = func
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, x: torch.Tensor):
        if isinstance(self.func, DEQFixedPointFunc):
            hidden_dim = self.func.linear_layer.out_features
        else:
            raise ValueError("func must be an instance of DEQFixedPointFunc or have a defined hidden_dim")

        z0 = torch.zeros(x.shape[0], hidden_dim, device=x.device)
        z_star = FixedPointSolver.apply(self.func, z0, x, self.max_iter, self.tol)
        return z_star
