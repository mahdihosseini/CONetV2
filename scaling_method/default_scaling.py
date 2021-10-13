
def round_even(number):
    return int(round(number / 2) * 2)
def default_scaling(metric_type, metric_difference, prev_channel_size,delta_scale = 1, max_conv_size = 256, min_conv_size = 16):
    positive_correlation = ['mQC']
    negative_correlation = ['newQC']
    scaling_factor = 1 + metric_difference * delta_scale
    if scaling_factor < 0.5:
        scaling_factor = 0.5
    elif scaling_factor > 2:
        scaling_factor = 2
    # Check if correlation is positive or negative
    # if metric_type in negative_correlation:
    #     scaling_factor = 1/scaling_factor
    new_channel_size = round_even(prev_channel_size * scaling_factor)
    # Make sure that we do not surpass the max and min conv sizes
    if new_channel_size > max_conv_size:
        new_channel_size = max_conv_size
    elif new_channel_size < min_conv_size:
        new_channel_size = min_conv_size
    return new_channel_size
