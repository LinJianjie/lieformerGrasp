import functools
import inspect
import sys
import warnings
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
string_types = (type(b''), type(u''))
import torch


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


def batchIndexing(input_xyz_query, input_xyz_list, batch_index, n_sample=1):
    results = []
    for i, input_xyz in enumerate(input_xyz_list):
        if input_xyz.dim() == 2:
            B, N = input_xyz.shape
        if input_xyz.dim() == 3:
            B, N, C = input_xyz.shape
        if input_xyz.dim() == 4:
            B, N, C1, C2 = input_xyz.shape
        _, S, _ = input_xyz_query.shape
        if batch_index.dim() == 2:
            batch_index = batch_index.unsqueeze(-1)
        idx_base = torch.arange(0, B, device=input_xyz.device).view(-1, 1, 1) * N
        idx = batch_index + idx_base
        idx = idx.reshape(-1)
        if input_xyz.dim() == 2:
            new_x = input_xyz.reshape(B * N, 1)[idx, :]
            select_x = new_x.view(B, S, n_sample)
        if input_xyz.dim() == 3:
            new_x = input_xyz.reshape(B * N, C)[idx, :]
            select_x = new_x.view(B, S, n_sample, C)
        if input_xyz.dim() == 4:
            new_x = input_xyz.reshape(B * N, C1, C2)[idx, :]
            select_x = new_x.view(B, S, n_sample, C1, C2)
        results.append(select_x.squeeze())
    return results


if __name__ == '__main__':
    xyz_1 = torch.rand(8, 10, 3)
    xyz_2 = torch.rand(8, 20, 3)
    xyz_3 = torch.rand(8, 20, 4, 4)
    dis = torch.cdist(xyz_1, xyz_2, p=2)
    min, index = torch.min(dis, dim=-1)
    aus_selected = batchIndexing(input_xyz_query=xyz_1, input_xyz=xyz_2, batch_index=index, n_sample=1)
    print(aus_selected.shape)
