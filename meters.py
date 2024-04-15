'''
Author: Souvik Kundu
'''

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg

class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.
    Args:
        name (str): the name of meter
    """
    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.reset()

    def reset(self):
        self.values = []

    def cache(self, value, pstep=1):
        self.steps += pstep
        self.values.append(value)

    def cache_list(self, value_list, pstep=1):
        self.steps += pstep
        self.values += value_list

    def flush(self, value, reset=True):
        pass


class ScalarMeter(Meter):
    """ScalarMeter records scalar over steps.
    """
    def __init__(self, name):
        super(ScalarMeter, self).__init__(name)

    def flush(self, value, step=-1, reset=True):
        if reset:
            self.reset()


def flush_scalar_meters(meters, method='avg'):
    """Docstring for flush_scalar_meters"""
    results = {}
    assert isinstance(meters, dict), "meters should be a dict."
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            continue
        if method == 'avg':
            value = sum(meter.values) / len(meter.values)
        elif method == 'sum':
            value = sum(meter.values)
        elif method == 'max':
            value = max(meter.values)
        elif method == 'min':
            value = min(meter.values)
        else:
            raise NotImplementedError(
                'flush method: {} is not yet implemented.'.format(method))
        results[name] = value
        meter.flush(value)
    return results
