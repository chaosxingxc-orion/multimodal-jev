"""Select equal TRAIN exposure for a bounded Stage1 capacity comparison."""
from .slurp_training import content_hash


def bring_pressure_forward(batches, global_batch, pressure):
    """Use one fixed real TRAIN batch as the second finite update."""
    offset, witness = pressure['offset'], pressure['anchor']
    if type(offset) is not int or offset % global_batch or offset < 2 * global_batch:
        raise ValueError('pressure offset must select a distinct complete TRAIN batch')
    index = offset // global_batch
    if index >= len(batches):
        raise ValueError('pressure batch is outside the source schedule')
    batch = batches[index]
    if batch['epoch'] != 0 or batch['real_queries'] != global_batch or not any(
            slot['id'] == witness for slot in batch['slots']):
        raise ValueError('the pressure witness is not in the pinned full TRAIN batch')
    batches[1], batches[index] = batch, batches[1]


def select_batches(batches, global_batch, probe, stop_update):
    prefix = []
    used = set()
    for span in probe['spans']:
        offset, count = span['offset'], span['recordings']
        if (type(offset) is not int or type(count) is not int or offset < 0 or count <= 0
                or offset % global_batch or count % global_batch):
            raise ValueError('each comparison span must contain complete batches for both widths')
        selected = batches[offset // global_batch:(offset + count) // global_batch]
        if (len(selected) * global_batch != count or
                any(b['epoch'] != 0 or b['real_queries'] != global_batch for b in selected)):
            raise ValueError('comparison spans require real epoch-zero TRAIN anchors')
        ids = [slot['id'] for batch in selected for slot in batch['slots']]
        if len(set(ids)) != count or used.intersection(ids):
            raise ValueError('comparison spans must be disjoint')
        if span.get('anchor_sha256', content_hash(ids)) != content_hash(ids):
            raise ValueError('comparison anchor order differs from the preflight')
        used.update(ids)
        prefix.extend(selected)
    if len(prefix) != stop_update:
        raise ValueError('comparison span budget differs from the trainer window')
    return prefix + batches[len(prefix):]
