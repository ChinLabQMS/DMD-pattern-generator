from typing import Dict, List
import json

from .painter import Painter
from .frame import ColorFrame, BinarySequence
from .utils import CompactJSONEncoder


class Pattern:
    def __init__(self, 
                 directory: str='', 
                 filename: str='', 
                 description: str='', 
                 params: Dict[str, any]={},
                 **kwargs):
        self.directory = directory
        self.filename = filename[:-4] if filename.endswith('.bmp') else filename
        self.description = description
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.create()
    
    def __str__(self):
        return f'{self.__class__.__name__}: {self.description}\n{self.directory}/{self.filename}\n{self.params}'
    
    def __repr__(self):
        return self.__str__()

    def create(self):
        raise NotImplementedError

    def apply(self, param, reset=False):
        raise NotImplementedError
    
    def display(self):
        raise NotImplementedError
    
    def save(self, save_config=True, filter_keys=()):
        if save_config:
            config = {k: v for k, v in self.__dict__.items() if k not in filter_keys}
            with open(f'{self.directory}/{self.filename}.json', 'w') as f:
                json.dump(config, f, indent=4, cls=CompactJSONEncoder)


class ColorPattern(Pattern):
    def __init__(self, 
                 directory: str='', 
                 filename: str='', 
                 description: str='', 
                 params: List[Dict[str, any]]=(),
                 **kwargs):
        super().__init__(directory, filename, description, params, **kwargs)
    
    def create(self):
        self.frame = ColorFrame()
        if len(self.params) > 0:
            self.apply(self.params[0], reset=True)
        for param in self.params[1:]:
            self.apply(param)
    
    def apply(self, param, reset=False):
        if 'painter_method' not in param:
            corr = None
        else:
            painter = Painter()
            corr = getattr(painter, param['painter_method'])(**param['painter_args'])
        self.frame.drawPattern(corr=corr, reset=reset, **param['draw_args'])

    def display(self, pattern_name=''):
        self.frame.displayPattern(dmd_space_title='DMD-space: ' + pattern_name,
                                  real_space_title='Real-space: ' + pattern_name)

    def save(self, save_config=True):
        self.frame.saveFrameToFile(self.directory, self.filename)
        super().save(save_config, filter_keys=['frame'])


class SequencePattern(Pattern):
    def __init__(self, 
                 directory: str='', 
                 filename: str='', 
                 description: str='',
                 num_frames: int=24,
                 params: List[Dict[str, any]]=(),
                 **kwargs):
        super().__init__(directory, filename, description, params, 
                         num_frames=num_frames, **kwargs)

    def create(self):
        self.sequence = BinarySequence(nframes=self.num_frames)
        if len(self.params) > 0:
            self.apply(self.params[0], reset=True)
        for param in self.params[1:]:
            self.apply(param)

    def apply(self, param, reset=False):
        var_name, var_values = param['sequence_varname'], param['sequence_varvalues']
        assert len(var_values) <= self.num_frames, f'Number of frames ({self.num_frames}) is less than number of values ({len(var_values)})'
        if len(var_values) < self.num_frames:
            var_values += [var_values[-1]] * (self.num_frames - len(var_values))
        painter = Painter()
        for i, var in enumerate(var_values):
            param['painter_args'].update({var_name: var})
            corr = getattr(painter, param['painter_method'])(**param['painter_args'])
            self.sequence.drawPatternOnFrame(i, corr=corr, reset=reset, **param['draw_args'])
    
    def display(self):
        self.sequence.displayBinaryFrames()

    def save(self, save_config=True, save_gif=True):
        self.sequence.saveRGBFrames(self.directory, self.filename)
        if save_gif:
            self.sequence.saveSequenceToGIF(self.directory, self.filename)
        if save_config:
            config = {k: v for k, v in self.__dict__.items() if k not in ['sequence']}
            with open(f'{self.directory}/{self.filename}.json', 'w') as f:
                json.dump(config, f, indent=4, cls=CompactJSONEncoder)
