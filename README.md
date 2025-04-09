# CPLUS Core Library for CPLUS QGIS plugin

CPLUS Core is the core module for [CPLUS plugin](https://github.com/ConservationInternational/cplus-plugin) and [CPLUS API](https://github.com/kartoza/cplus-api). This repository houses the core logic for the analysis, encapsulated as a QGIS task, which is utilized by the CPLUS plugin in QGIS.

### Example Usage

```
    # create task_config object
    analysis_config = TaskConfig(
        self.scenario,
        self.task_config.priority_layers,
        self.scenario.priority_layer_groups,
        self.scenario.activities,
        ...
    )

    # create analysis task
    self.analysis_task = ScenarioAnalysisTask(analysis_config)

    # setup signals
    self.analysis_task.custom_progress_changed.connect(
        self.set_custom_progress)
    self.analysis_task.status_message_changed.connect(
        self.set_status_message)
    self.analysis_task.info_message_changed.connect(self.set_info_message)
    self.analysis_task.log_received.connect(self.log_message)
    self.analysis_task.task_cancelled.connect(self.cancel_task)

    # call run
    self.analysis_task.run()
```


### 📃 Documentation

The plugin documentation page is available at https://ConservationInternational.github.io/cplus-plugin
