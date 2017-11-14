def interpolated_decay(start_lr, end_lr, global_step, start_step, end_step,
                    name=None):
      """ TODO explain
      """
    if global_step is None:
        raise ValueError("global_step is required for interpolated_decay.")
    with ops.name_scope(name, "PolynomialDecay", [start_lr, end_lr, global_step, 
            start_step, end_step]) as name:
        start_lr = ops.convert_to_tensor(start_lr, name="start_lr")
        dtype = learning_rate.dtype
        end_lr = math_ops.cast(end_learning_rate, dtype)
        global_step = math_ops.cast(global_step, dtype)
        start_step = math_ops.cast(start_step, dtype)
        end_step = math_ops.cast(end_step, dtype)
        
        # Make sure that the global_step used is not bigger than decay_steps.
        global_step = math_ops.minimum(global_step, end_step)
        global_step = math_ops.maximum(global_step, start_step)
        global_step = math_ops.subtract(global_step, start_step)

        decay_steps = math_ops.subtract(end_step, start_step)
        p = math_ops.div(global_step, decay_steps)
        
        return math_ops.add(math_ops.multiply(start_lr - end_lr, 1 - p),
                            end_lr, name=name)
