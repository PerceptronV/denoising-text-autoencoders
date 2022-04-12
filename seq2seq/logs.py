from email.policy import default
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, dir, step=0, default_path="Log") -> None:
        self.logdir = dir
        self.gstep = step
        self.default_path = default_path
        self.writer = SummaryWriter(dir)
    
    def __call__(self, value, path=None, *args, **kwargs):
        if path is None:
            path = self.default_path

        print(f"{path}: {value}")

        if type(value).__name__ in ("int", "float"):
            self.write_scalar(path, value, *args, **kwargs)
        elif type(value).__name__ == "str":
            self.write_text(path, value, *args, **kwargs)
        else:
            raise TypeError(f"Unsupported type: {type(value).__name__}")
    
    def set_global_step(self, step_val):
        self.gstep = step_val
    
    def solve_step_inc(self, step, inc):
        cur_step = self.gstep

        if inc is None or inc == True:
            self.gstep += 1
        
        if step is None:
            return cur_step
        
        return step
    
    def write_text(self, path, value, step=None, increment=False):
        step = self.solve_step_inc(step, increment)
        self.writer.add_text(path, value, step)

    def write_scalar(self, path, value, step=None, increment=True):
        step = self.solve_step_inc(step, increment)
        self.writer.add_scalar(path, value, step)
