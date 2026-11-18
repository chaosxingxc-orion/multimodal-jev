"""Derive process-visible ordinals from the container's sorted physical set."""


def visibility(manifest):
    train=manifest['training_devices'];services=manifest['inference_services']
    devices=[*train,*[s['device'] for s in services]]
    if (not train or not services or any(type(d) is not int or d<0 for d in devices)
            or len(set(devices))!=len(devices)
            or len({s['name'] for s in services})!=len(services)):
        raise ValueError('distinct declared physical devices and named services required')
    mounted=sorted(devices)
    return (','.join(str(mounted.index(d)) for d in train),
            [(s['name'],str(mounted.index(s['device']))) for s in services])
