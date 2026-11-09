"""Effective-token accounting, separate from FLOPs and optimizer steps."""
FIELDS=('sequences','prompt_tokens','response_tokens','audio_frames',
        'gradient_sequences','gradient_tokens','no_grad_sequences','no_grad_tokens',
        'optimizer_steps')


def batch_budget(prompt_lengths,response_lengths,audio_lengths,*,gradient_passes,
                 no_grad_passes,optimizer_steps):
    lengths=[list(values) for values in (prompt_lengths,response_lengths,audio_lengths)]
    size=len(lengths[0])
    if not size or any(len(values)!=size for values in lengths):
        raise ValueError('aligned nonempty budget lengths required')
    if any(type(n) is not int or n<1 for values in lengths for n in values):
        raise ValueError('positive integer effective lengths required')
    if any(type(n) is not int or n<0 for n in (gradient_passes,no_grad_passes,optimizer_steps)):
        raise ValueError('nonnegative integer execution counts required')
    prompt,response,audio=map(sum,lengths)
    return dict(sequences=size,prompt_tokens=prompt,response_tokens=response,audio_frames=audio,
                gradient_sequences=size*gradient_passes,gradient_tokens=(prompt+response)*gradient_passes,
                no_grad_sequences=size*no_grad_passes,no_grad_tokens=(prompt+response)*no_grad_passes,
                optimizer_steps=optimizer_steps)


def sum_budgets(budgets):
    total=dict.fromkeys(FIELDS,0)
    for budget in budgets:
        if set(budget)!=set(FIELDS):raise ValueError('incomplete or unknown budget fields')
        for key,value in budget.items():
            if type(value) is not int or value<0:raise ValueError('nonnegative integer budget counts required')
            total[key]+=value
    return total
