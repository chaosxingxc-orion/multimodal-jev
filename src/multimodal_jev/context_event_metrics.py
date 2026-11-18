"""Complete exclusive-event evaluation with fixed candidate order and threshold."""
import math

from .candidate_set_metrics import CandidateSetMetrics


class ContextEventMetrics:
    def __init__(self,candidate_ids,*,threshold=.5):
        self.ids=tuple(candidate_ids)
        if (not self.ids or len(set(self.ids))!=len(self.ids) or
                any(type(x) is not str or not x for x in self.ids)):
            raise ValueError('fixed distinct candidate order required')
        self.set_metrics=CandidateSetMetrics(threshold)
        self.confusion={x:{y:0 for y in self.ids} for x in self.ids}

    def add(self,query_id,gold,probabilities):
        if gold not in self.ids or len(probabilities)!=len(self.ids):
            raise ValueError('known exclusive event and complete probability vector required')
        mapping=dict(zip(self.ids,probabilities,strict=True))
        self.set_metrics.add(query_id,{k:k==gold for k in self.ids},mapping)
        # Stable ontology order resolves exact ties; report the rule explicitly.
        selected=max(self.ids,key=lambda k:mapping[k])
        self.confusion[gold][selected]+=1

    def result(self):
        base=self.set_metrics.result();n=base['counts']['queries']
        classes=[]
        for k in self.ids:
            tp=self.confusion[k][k];support=sum(self.confusion[k].values())
            returned=sum(row[k] for row in self.confusion.values())
            denom=support+returned
            classes.append(dict(id=k,support=support,predicted=returned,tp=tp,
                f1=2*tp/denom if denom else 0.))
        return dict(queries=n,candidate_ids=list(self.ids),threshold=self.set_metrics.threshold,
            query_macro_set_f1=base['macro_query_observed']['f1'],
            set_precision=base['macro_query_observed']['precision'],
            set_precision_defined_queries=base['counts']['precision_queries'],
            set_recall=base['macro_query_observed']['recall'],
            exact_set=base['exact_set_accuracy_complete'],mean_selected_count=base['mean_returned_candidates'],
            top1_event_accuracy=sum(self.confusion[k][k] for k in self.ids)/n,
            # Recorded curves were produced with compensated summation; plain
            # left-to-right sums can differ by one ULP and fail exact re-audit.
            top1_macro_f1=math.fsum(r['f1'] for r in classes)/len(classes),
            candidate_binary_nll=base['mean_binary_nll_per_observed_query'],
            binary_nll_infinite=base['binary_nll_infinite'],candidate_brier=base['mean_binary_brier_per_observed_query'],
            classes=classes,confusion=self.confusion,top1_tie_rule='first_in_fixed_candidate_order',
            scope='Exclusive benchmark event identity; independent sigmoid sets at fixed threshold, not semantic intent presence.')
