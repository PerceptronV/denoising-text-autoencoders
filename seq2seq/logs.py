from email.policy import default
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, dir, step=0, default_path="Log") -> None:
        self.logdir = dir
        self.gstep = step
        self.default_path = default_path
        self.writer = SummaryWriter(dir)

        self.buffer = ""
        self.bufferStep = step
    
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

        if path != self.default_path:
            self.writer.add_text(path, value, step)
        
        else:
            if step != self.bufferStep:
                if self.buffer != '':
                    self.writer.add_text(path, self.buffer, self.bufferStep)
                self.buffer = value
                self.bufferStep = step
            else:
                if self.buffer != '':
                    self.buffer = self.buffer + '\n' + value
                else:
                    self.buffer = value            
    
    def flush(self, path=None):
        if path is None:
            path = self.default_path
        self.writer.add_text(path, self.buffer, self.gstep)

    def write_scalar(self, path, value, step=None, increment=True):
        step = self.solve_step_inc(step, increment)
        self.writer.add_scalar(path, value, step)
