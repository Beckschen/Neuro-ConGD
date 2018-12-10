import numpy as np
from tqdm import tqdm


class Generator:
    def num_batches(self):
        pass

    def iterate(self):
        pass


class Model:
    def __init__(self, inputs, outputs, initial_state, final_state, seq_lengths, labels):
        self.inputs = inputs
        self.outputs = outputs
        self.initial_state = initial_state
        self.final_state = final_state
        self.seq_lengths = seq_lengths
        self.labels = labels


class Metric:
    def __init__(self):
        self.name = "metric"

    def fetches(self):
        return {}

    def value(self, values):
        pass

    def reduce(self, values, seq_lengths):
        return sum(values)

class TensorMetric(Metric):
    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor

    def fetches(self):
        return {self.name: self.tensor}

    def value(self, values):
        return values[self.name]

class AccuracyMetric(TensorMetric):
    def reduce(self, values, seq_lengths):
        return sum(values) / np.sum(seq_lengths)


class Callback:
    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def before_chunk(self):
        pass

    def after_chunk(self):
        pass


class Runner:
    def __init__(self, metrics=[], callbacks=[]):
        self.metrics = metrics
        self.callbacks = callbacks


    def run(self, sess, model, generator, chunk_size, extra_feeds={}, extra_fetches={}):
        fetches = {"final_state": model.final_state}
        fetches.update(extra_fetches)
        for m in self.metrics:
            fetches.update(m.fetches())

        # Request data from generator
        iterator = generator.iterate()

        # Keep a running total of all metrics and the number of batches
        # processed to compute a running average
        metric_values = {m: 0.0 for m in self.metrics}
        nbatches = 0

        batches = tqdm(range(generator.num_batches()))
        for batch in batches:
            seq_lengths, data, labels = next(iterator)

            # The training data is split into fixed length chunks so that
            # sequences of arbitrary length can be handled
            chunk_state = None
            offset = 0

            # Accumulate values over chunks before reducing them
            batch_metric_values = {m: [] for m in self.metrics}
            original_seq_lengths = seq_lengths.copy()

            # The loop always runs at least once, even if all sequences are
            # empty, so that the final states are properly set
            while np.any(seq_lengths > 0) or offset == 0:
                chunk_lengths = np.minimum(chunk_size, seq_lengths)
                chunk_data = data[:, offset:offset + max(chunk_lengths)]
                chunk_labels = labels[:, offset:offset + max(chunk_lengths)]

                if chunk_state is None:
                    feeds = {model.inputs: chunk_data,
                             model.seq_lengths: chunk_lengths,
                             model.labels: chunk_labels}
                else:
                    chunk_filter = chunk_lengths > 0
                    feeds = {model.inputs: chunk_data[chunk_filter],
                             model.seq_lengths: chunk_lengths[chunk_filter],
                             model.labels: chunk_labels[chunk_filter],
                             model.initial_state: chunk_state[chunk_filter]}

                feeds.update(extra_feeds)

                values = sess.run(fetches, feeds)

                if chunk_state is None:
                    chunk_state = values["final_state"]
                else:
                    chunk_state[chunk_filter] = values["final_state"]

                offset += chunk_size
                seq_lengths = np.maximum(0, seq_lengths - chunk_size)

                for m in self.metrics:
                    batch_metric_values[m].append(m.value(values))

            for m in self.metrics:
                metric_values[m] += m.reduce(batch_metric_values[m], original_seq_lengths)

            nbatches += 1

            desc = [f"{m.name} {v / nbatches:.3f}" for m, v in metric_values.items()]
            batches.set_description(", ".join(desc))
