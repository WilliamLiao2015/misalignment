from .longitudinal import is_accelerating, is_cruising, is_decelerating, is_standing_still
from .lateral import is_turning_right, is_turning_left

tests = {
    "longitudinal:driving-forward:accelerating": is_accelerating,
    "longitudinal:driving-forward:cruising": is_cruising,
    "longitudinal:driving-forward:decelerating": is_decelerating,
    "longitudinal:standing-still": is_standing_still,
    "lateral:turning:left": is_turning_left,
    "lateral:turning:right": is_turning_right
}
