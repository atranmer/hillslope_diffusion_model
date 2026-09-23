from Diffusion_Model import calculate_stable_timestep


def test_time_step_is_float():
    time_step = calculate_stable_timestep(1,1)
    assert isinstance(time_step, float)
