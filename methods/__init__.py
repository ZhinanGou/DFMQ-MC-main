from .MULT.manager import MULT_Manager
from .MAG_BERT.manager import MAG_BERT_Manager
from .MISA.manager import MISA_Manager
from .MMIM.manager import MMIM_Manager
from .TCL_MAP.manager import TCL_MAP_Manager
from .SDIF.manager import SDIF_Manager
from .MIntOOD.manager import MIntOOD_Manager
from .MCWP.manager import MCWP_Manager
method_map = {
    'mult': MULT_Manager,
    'mag_bert': MAG_BERT_Manager,
    'misa': MISA_Manager,
    'mmim': MMIM_Manager,
    'tcl_map': TCL_MAP_Manager,
    'sdif': SDIF_Manager,
    'mintood': MIntOOD_Manager,
    'mcwp':MCWP_Manager,
}