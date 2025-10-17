from omegaconf import OmegaConf


class Logger:
    def __init__(self):
        self.history = {}

    # def update(self, metric, value, stepIdx=None, epochIdx=None):
    #     assert stepIdx or epochIdx
    #     self.safeupdate(self.history, {metric: []})
    #     record = {'value': value}
    #     if stepIdx:
    #         record.update({'step': stepIdx})
    #     if epochIdx:
    #         record.update({'epoch': epochIdx})
    #     self.history[metric].append(record)
    #     self.safeupdate(self.history, {'step_history': []})
    #     record = {metric: value, 'epoch': epochIdx} if epochIdx else {metric: value}
    #     self.history['step_history'].append({stepIdx: {metric: value, 'epoch': epochIdx}})
    #     self.safeupdate(self.history, {'epoch_history': {}})
    #     self.safeupdate(self.history['epoch_history'], {epochIdx: []})
    #     record = {metric: value, 'step': stepIdx} if stepIdx else {metric: value}
    #     self.history['epoch_history'][epochIdx].append(record)


    def update(self, metric, value, stepIdx=None, epochIdx=None):
        assert stepIdx is not None or epochIdx is not None
        self.safeupdate(self.history, {metric: []})
        metricRecord = {'value': value}
        if stepIdx:
            metricRecord.update({'step': stepIdx})
            self.safeupdate(self.history, {'step_history': []})
            record = {metric: value, 'epoch': epochIdx} if epochIdx else {metric: value}
            self.history['step_history'].append({stepIdx: {metric: value, 'epoch': epochIdx}})
        if epochIdx:
            metricRecord.update({'epoch': epochIdx})
            self.safeupdate(self.history, {'epoch_history': {}})
            self.safeupdate(self.history['epoch_history'], {epochIdx: []})
            record = {metric: value, 'step': stepIdx} if stepIdx else {metric: value}
            self.history['epoch_history'][epochIdx].append(record)
        self.history[metric].append(metricRecord)

    @staticmethod
    def safeupdate(root: dict, leaves: dict):
        root.update({k: v for k, v in leaves.items() if k not in root.keys()})

    def save(self, path: str):
        history = OmegaConf.create({k: self.history[k] for k in sorted(self.history.keys())})
        OmegaConf.save(history, path)


if __name__ == '__main__':
    logger = Logger()
    logger.save('tmp1.yml')
    logger = Logger()
    logger.safeupdate(logger.history, {'This is a test': 'This is a test'})
    logger.save('tmp2.yml')
    logger = Logger()
    logger.update('testmetric1', 'testvalue1', 'teststep1', 'testepoch1')
    logger.update('testmetric2', 'testvalue2', 'teststep2', 'testepoch2')
    logger.update('testmetric3', 'testvalue3', 'teststep1', 'testepoch2')
    logger.update('testmetric4', 'testvalue4', 'teststep2', 'testepoch1')
    logger.save('tmp3.yml')