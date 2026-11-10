from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from check_sfc_omni_long_audio import expected_mask_frames


def test_frame_mask_accounts_for_final_stft_frame_and_shorter_clip_padding():
    assert expected_mask_frames([480001],160)==[3000]
    assert expected_mask_frames([480001,32001],160)==[3000,201]
    assert expected_mask_frames([480160,32000],160)==[3001,200]
    assert expected_mask_frames([32001,480001],160)==[201,3000]
