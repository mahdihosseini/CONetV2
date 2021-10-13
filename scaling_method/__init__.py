from scaling_method.default_scaling import default_scaling

def getScalingMethod(method_name, **kwargs):
    scaling_methods = {
        'default': default_scaling
    }
    return scaling_methods[method_name](**kwargs)