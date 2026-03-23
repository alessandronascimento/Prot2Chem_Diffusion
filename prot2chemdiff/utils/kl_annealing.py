class KLAnnealer:
    def __init__(self, total_steps, max_beta=0.002):
        self.total_steps = total_steps
        self.max_beta = max_beta
        self.current_step = 0

    def step(self):
        self.current_step += 1
        warmup = self.total_steps * 0.25
        if self.current_step < warmup:
            return self.max_beta * (self.current_step / warmup)
        return self.max_beta