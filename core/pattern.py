from typing import Dict
import json

from .painter import Painter
from .frame import ColorFrame
from .utils import CompactJSONEncoder


class CalibPattern:
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
        return f'Pattern object: {self.directory}/{self.filename}, {self.description}'
    
    def __repr__(self):
        return f'Pattern object: {self.directory}/{self.filename}, {self.description}'
    
    def create(self):
        self.frame = ColorFrame()
        self.painter = Painter()
        if len(self.params) > 0:
            self.apply(self.params[0], reset=True)
        for param in self.params[1:]:
            self.apply(param)
    
    def apply(self, param, reset=False):
        if 'painter_method' not in param:
            corr = None
        else:
            painter_method = getattr(self.painter, param['painter_method'])
            corr = painter_method(**param['painter_args'])
        self.frame.drawPattern(corr=corr, reset=reset, **param['draw_args'])

    def display(self, pattern_name=''):
        self.frame.displayPattern(dmd_space_title='DMD-space: ' + pattern_name,
                                  real_space_title='Real-space: ' + pattern_name)

    def save(self, save_config=True):
        self.frame.saveFrameToFile(self.directory, self.filename)
        if save_config:
            config = {k: v for k, v in self.__dict__.items() if k not in ['frame', 'painter']}
            with open(f'{self.directory}/{self.filename}.json', 'w') as f:
                json.dump(config, f, indent=4, cls=CompactJSONEncoder)
