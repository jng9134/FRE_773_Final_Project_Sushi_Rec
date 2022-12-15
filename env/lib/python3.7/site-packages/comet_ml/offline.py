# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

"""
Author: Boris Feld

This module contains the code related to offline feature

"""
import io
import json
import logging
import os
import os.path
import shutil
import tempfile
import traceback
import zipfile
from os.path import join
from zipfile import ZipFile

from jsonschema import ValidationError

from ._reporting import (
    EXPERIMENT_CREATED,
    OFFLINE_INVALID_UPLOAD_MSG,
    OFFLINE_INVALID_WS_MSG,
)
from ._typing import Optional
from .comet import OfflineStreamer, format_url, is_valid_experiment_key
from .compat_utils import json_dump
from .config import (
    ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    DEFAULT_OFFLINE_DATA_DIRECTORY,
    DEFAULT_WS_JOIN_TIMEOUT,
    get_api_key,
    get_config,
    get_previous_experiment,
)
from .connection import (
    FileUploadManager,
    FileUploadManagerMonitor,
    RestServerConnection,
    WebSocketConnection,
    format_messages_for_ws,
    get_backend_address,
    get_rest_api_client,
)
from .exceptions import (
    CometRestApiException,
    ExperimentAlreadyUploaded,
    InvalidAPIKey,
    InvalidOfflineDirectory,
)
from .experiment import BaseExperiment
from .feature_toggles import FeatureToggles
from .logging_messages import (
    CLOUD_DETAILS_MSG_SENDING_ERROR,
    FILE_UPLOADS_PROMPT,
    MODEL_GRAPH_MSG_SENDING_ERROR,
    OFFLINE_DATA_DIR_DEFAULT_WARNING,
    OFFLINE_DATA_DIR_FAILED_WARNING,
    OFFLINE_EXPERIMENT_ALREADY_UPLOADED,
    OFFLINE_EXPERIMENT_END,
    OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG,
    OFFLINE_EXPERIMENT_INVALID_WS_MSG,
    OFFLINE_SENDER_ENDS,
    OFFLINE_SENDER_STARTS,
    OS_PACKAGE_MSG_SENDING_ERROR,
    WAITING_DATA_UPLOADED,
)
from .metrics import MetricsSampler
from .schemas import (
    get_cloud_details_msg_validator,
    get_experiment_file_validator,
    get_graph_msg_validator,
    get_os_packages_msg_validator,
    get_remote_file_msg_validator,
    get_system_details_msg_validator,
    get_upload_msg_validator,
    get_ws_msg_validator,
)
from .utils import generate_guid, local_timestamp, wait_for_done

LOGGER = logging.getLogger(__name__)


class OfflineExperiment(BaseExperiment):
    def __init__(
        self,
        project_name=None,  # type: Optional[str]
        workspace=None,  # type: Optional[str]
        log_code=True,  # type: Optional[bool]
        log_graph=True,  # type: Optional[bool]
        auto_param_logging=True,  # type: Optional[bool]
        auto_metric_logging=True,  # type: Optional[bool]
        parse_args=True,  # type: Optional[bool]
        auto_output_logging="default",  # type: Optional[str]
        log_env_details=True,  # type: Optional[bool]
        log_git_metadata=True,  # type: Optional[bool]
        log_git_patch=True,  # type: Optional[bool]
        disabled=False,  # type: Optional[bool]
        offline_directory=None,  # type: Optional[str]
        log_env_gpu=True,  # type: Optional[bool]
        log_env_host=True,  # type: Optional[bool]
        api_key=None,  # type: Optional[str]
        display_summary=None,  # type: Optional[bool]
        log_env_cpu=True,  # type: Optional[bool]
        display_summary_level=None,  # type: Optional[int]
        auto_weight_logging=None,  # type: Optional[bool]
        auto_log_co2=False,  # type: Optional[bool]
        auto_metric_step_rate=10,  # type: Optional[int]
        auto_histogram_tensorboard_logging=False,  # type: Optional[bool]
        auto_histogram_epoch_rate=1,  # type: Optional[int]
        auto_histogram_weight_logging=False,  # type: Optional[bool]
        auto_histogram_gradient_logging=False,  # type: Optional[bool]
        auto_histogram_activation_logging=False,  # type: Optional[bool]
    ):
        # type: (...) -> None
        """
        Creates a new experiment and serialize it on disk. The experiment file will need to be
        upload manually later to appears on the frontend.
        Args:
            project_name: Optional. Send your experiment to a specific project. Otherwise will be sent to `Uncategorized Experiments`.
                             If project name does not already exists Comet.ml will create a new project.
            workspace: Optional. Attach an experiment to a project that belongs to this workspace
            log_code: Default(True) - allows you to enable/disable code logging
            log_graph: Default(True) - allows you to enable/disable automatic computation graph logging.
            auto_param_logging: Default(True) - allows you to enable/disable hyper parameters logging
            auto_metric_logging: Default(True) - allows you to enable/disable metrics logging
            auto_metric_step_rate: Default(10) - controls how often batch metrics are logged
            auto_histogram_tensorboard_logging: Default(False) - allows you to enable/disable automatic histogram logging
            auto_histogram_epoch_rate: Default(1) - controls how often histograms are logged
            auto_histogram_weight_logging: Default(False) - allows you to enable/disable automatic histogram logging of biases and weights
            auto_histogram_gradient_logging: Default(False) - allows you to enable/disable automatic histogram logging of gradients
            auto_histogram_activation_logging: Default(False) - allows you to enable/disable automatic histogram logging of activations
            auto_output_logging: Default("default") - allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"default"` which will detect your environment and deactivate
                the output logging for IPython and Jupyter environment and
                sets `"native"` in the other cases.
            auto_log_co2: Default(True) - automatically tracks the CO2 emission of
                this experiment if `codecarbon` package is installed in the environment
            parse_args: Default(True) - allows you to enable/disable automatic parsing of CLI arguments
            log_env_details: Default(True) - log various environment
                information in order to identify where the script is running
            log_env_gpu: Default(True) - allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu: Default(True) - allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_host: Default(True) - allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata: Default(True) - allow you to enable/disable the
                automatic collection of git details
            display_summary_level: Default(1) - control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled: Default(False) - allows you to disable all network
                communication with the Comet.ml backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
            offline_directory: the directory used to save the offline archive
                for the experiment.
        """
        self.config = get_config()

        self.api_key = get_api_key(
            api_key, self.config
        )  # optional, except for on-line operations

        # Start and ends time
        self.start_time = None
        self.stop_time = None
        self.mode = "create"

        super(OfflineExperiment, self).__init__(
            project_name=project_name,
            workspace=workspace,
            log_code=log_code,
            log_graph=log_graph,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            auto_output_logging=auto_output_logging,
            log_env_details=log_env_details,
            log_git_metadata=log_git_metadata,
            log_git_patch=log_git_patch,
            disabled=disabled,
            log_env_gpu=log_env_gpu,
            log_env_host=log_env_host,
            display_summary=display_summary,
            display_summary_level=display_summary_level,
            log_env_cpu=log_env_cpu,
            auto_weight_logging=auto_weight_logging,
            auto_log_co2=auto_log_co2,
            auto_metric_step_rate=auto_metric_step_rate,
            auto_histogram_epoch_rate=auto_histogram_epoch_rate,
            auto_histogram_tensorboard_logging=auto_histogram_tensorboard_logging,
            auto_histogram_weight_logging=auto_histogram_weight_logging,
            auto_histogram_gradient_logging=auto_histogram_gradient_logging,
            auto_histogram_activation_logging=auto_histogram_activation_logging,
        )

        if not self.disabled:
            # Check that the offline directory is usable
            self.offline_directory, default_dir = self._get_offline_data_dir_path(
                offline_directory
            )
            # Try to create ZIP file for the experiment
            zip_file, self.offline_directory = self._create_offline_archive(
                fallback_to_temp=default_dir
            )
            # Close the file handle, it will be reopened later
            zip_file.close()

        if self.disabled is not True:
            if api_key is not None:
                self._log_once_at_level(
                    logging.WARNING,
                    "api_key was given, but is ignored in OfflineExperiment(); remember to set when you upload",
                )
            elif self.api_key is not None:
                self._log_once_at_level(
                    logging.INFO,
                    "COMET_API_KEY was set, but is ignored in OfflineExperiment(); remember to set when you upload",
                )

            self._start()

            if self.alive is True:
                self._report(event_name=EXPERIMENT_CREATED)

    def _create_offline_archive(self, fallback_to_temp=True):
        # type: (bool) -> (ZipFile, str)
        """Attempts to create ZIP file in the offline_directory. If attempt failed it will try to repeat in the
        temporary directory if fallback_to_temp is set. Otherwise, InvalidOfflineDirectory raised.
        Args:
            fallback_to_temp:
                The flag to indicate whether automatic recovery should be performed in case if failed to create archive
                in the current offline_directory.
        Returns:
            The ZipFile for offline archive and the path to the offline directory where archive was created
        """
        try:
            # create parent directory
            if not os.path.exists(self.offline_directory):
                os.mkdir(self.offline_directory, 0o700)

            # Try to create the archive now
            return (
                _create_zip_file(self.offline_directory, self.id),
                self.offline_directory,
            )
        except (OSError, IOError) as exc:
            if fallback_to_temp:
                # failed - use temporary directory instead
                offline_dir = tempfile.mkdtemp()
                LOGGER.warning(
                    OFFLINE_DATA_DIR_FAILED_WARNING,
                    os.path.abspath(self.offline_directory),
                    os.path.abspath(offline_dir),
                    str(exc),
                    exc_info=True,
                )
                return _create_zip_file(offline_dir, self.id), offline_dir
            else:
                raise InvalidOfflineDirectory(self.offline_directory, str(exc))

    def _get_offline_data_dir_path(self, offline_directory):
        # type: (Optional[str]) -> (str, bool)
        """looks for the offline data directory path. If offline_directory parameter is not None it will be returned.
        Otherwise, the experiment configuration settings will be used. Finally, if later failed the default data
        directory will be used.
        Args:
            offline_directory:
                The provided by user path to the offline directory or None if not set.
        Returns:
            The calculated path to the offline directory and flag to indicate if default data directory suffix was used.
        """
        if offline_directory is None:
            offline_directory = self.config["comet.offline_directory"]

        if (
            offline_directory is not None
            and offline_directory != DEFAULT_OFFLINE_DATA_DIRECTORY
        ):
            return offline_directory, False

        # use DEFAULT_OFFLINE_DATA_DIRECTORY in the CWD
        offline_directory = join(os.getcwd(), DEFAULT_OFFLINE_DATA_DIRECTORY)

        # display warning to the user that offline directory was chosen with default path
        LOGGER.info(OFFLINE_DATA_DIR_DEFAULT_WARNING, offline_directory)

        return offline_directory, True

    def display(self, *args, **kwargs):
        """Do nothing"""
        pass

    def display_project(self, *args, **kwargs):
        """Do nothing"""
        pass

    def _start(self, **kwargs):
        self.start_time = local_timestamp()
        super(OfflineExperiment, self)._start(**kwargs)
        self.log_other("offline_experiment", True)

    def _write_experiment_meta_file(self):
        meta_file_path = join(self.tmpdir, "experiment.json")
        meta = {
            "offline_id": self.id,
            "project_name": self.project_name,
            "start_time": self.start_time,
            "stop_time": self.stop_time,
            "tags": self.get_tags(),
            "workspace": self.workspace,
            "mode": self.mode,
        }
        with open(meta_file_path, "wb") as f:
            json_dump(meta, f)

    def _mark_as_ended(self):
        if not self.alive:
            LOGGER.debug("Skipping creating the offline archive as we are not alive")
            return

        LOGGER.info("Starting saving the offline archive")
        self.stop_time = local_timestamp()

        self._write_experiment_meta_file()

        zip_file, self.offline_directory = self._create_offline_archive(
            fallback_to_temp=True
        )

        for file in os.listdir(self.tmpdir):
            zip_file.write(os.path.join(self.tmpdir, file), file)

        zip_file.close()

        # Clean the tmpdir to avoid filling up the disk
        try:
            shutil.rmtree(self.tmpdir)
        except OSError:
            # We made our best effort to clean ourselves
            LOGGER.debug(
                "Error cleaning offline experiment tmpdir: %r",
                self.tmpdir,
                exc_info=True,
            )

        # Display the full command to upload the offline experiment
        LOGGER.info(OFFLINE_EXPERIMENT_END, zip_file.filename)

    def _setup_streamer(self):
        """
        Initialize the streamer and feature flags.
        """
        # Initiate the streamer
        self.streamer = OfflineStreamer(self.tmpdir, 0)

        # Start streamer thread.
        self.streamer.start()

        self.feature_toggles = FeatureToggles({}, self.config)

        # Mark the experiment as alive
        return True

    def _report(self, *args, **kwrags):
        # TODO WHAT TO DO WITH REPORTING?
        pass

    def _get_experiment_url(self, tab=None):
        return "[OfflineExperiment will get URL after upload]"


class ExistingOfflineExperiment(OfflineExperiment):
    def __init__(
        self,
        project_name=None,  # type: Optional[str]
        workspace=None,  # type: Optional[str]
        log_code=True,  # type: Optional[bool]
        log_graph=True,  # type: Optional[bool]
        auto_param_logging=True,  # type: Optional[bool]
        auto_metric_logging=True,  # type: Optional[bool]
        parse_args=True,  # type: Optional[bool]
        auto_output_logging="default",  # type: Optional[str]
        log_env_details=True,  # type: Optional[bool]
        log_git_metadata=True,  # type: Optional[bool]
        log_git_patch=True,  # type: Optional[bool]
        disabled=False,  # type: Optional[bool]
        offline_directory=None,  # type: Optional[str]
        log_env_gpu=True,  # type: Optional[bool]
        log_env_host=True,  # type: Optional[bool]
        api_key=None,  # type: Optional[str]
        display_summary=None,  # type: Optional[bool]
        log_env_cpu=True,  # type: Optional[bool]
        display_summary_level=None,  # type: Optional[int]
        auto_weight_logging=False,  # type: Optional[bool]
        previous_experiment=None,  # type: Optional[str]
    ):
        # type: (...) -> None
        """
        Continue a previous experiment (identified by previous_experment) and serialize it on disk.
        The experiment file will need to be upload manually later to append new information to the
        previous experiment. The previous experiment need to exists before upload of the
        ExistingOfflineExperiment.
        Args:
            previous_experiment: Optional. Your experiment key from comet.ml, could be set through
                configuration as well.
            project_name: Optional. Send your experiment to a specific project. Otherwise will be sent to `Uncategorized Experiments`.
                             If project name does not already exists Comet.ml will create a new project.
            workspace: Optional. Attach an experiment to a project that belongs to this workspace
            log_code: Default(True) - allows you to enable/disable code logging
            log_graph: Default(True) - allows you to enable/disable automatic computation graph logging.
            auto_param_logging: Default(True) - allows you to enable/disable hyper parameters logging
            auto_metric_logging: Default(True) - allows you to enable/disable metrics logging
            parse_args: Default(True) - allows you to enable/disable automatic parsing of CLI arguments
            auto_output_logging: Default("default") - allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"default"` which will detect your environment and deactivate
                the output logging for IPython and Jupyter environment and
                sets `"native"` in the other cases.
            log_env_details: Default(True) - log various environment
                information in order to identify where the script is running
            log_env_gpu: Default(True) - allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu: Default(True) - allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_host: Default(True) - allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata: Default(True) - allow you to enable/disable the
                automatic collection of git details
            display_summary_level: Default(1) - control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled: Default(False) - allows you to disable all network
                communication with the Comet.ml backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
            offline_directory: the directory used to save the offline archive
                for the experiment.
        """
        self.config = get_config()

        self.previous_experiment = get_previous_experiment(
            previous_experiment, self.config
        )

        if not is_valid_experiment_key(self.previous_experiment):
            raise ValueError("Invalid experiment key: %s" % self.previous_experiment)

        super(ExistingOfflineExperiment, self).__init__(
            project_name=project_name,
            workspace=workspace,
            log_code=log_code,
            log_graph=log_graph,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            auto_output_logging=auto_output_logging,
            log_env_details=log_env_details,
            log_git_metadata=log_git_metadata,
            log_git_patch=log_git_patch,
            disabled=disabled,
            offline_directory=offline_directory,
            log_env_gpu=log_env_gpu,
            log_env_host=log_env_host,
            api_key=api_key,
            log_env_cpu=log_env_cpu,
            display_summary=display_summary,
            display_summary_level=display_summary_level,
            auto_weight_logging=auto_weight_logging,
        )

        self.mode = "append"

    def _get_experiment_key(self):
        return self.previous_experiment


def unzip_offline_archive(offline_archive_path):
    temp_dir = tempfile.mkdtemp()

    zip_file = zipfile.ZipFile(offline_archive_path, mode="r", allowZip64=True)

    # Extract the archive
    zip_file.extractall(temp_dir)

    return temp_dir


class OfflineSender(object):
    def __init__(
        self,
        api_key,
        offline_dir,
        force_reupload=False,
        display_level="info",
        validation_error=False,
        file_upload_waiting_timeout=ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    ):
        self.config = get_config()
        self.api_key = api_key
        self.offline_dir = offline_dir
        self.force_reupload = force_reupload
        self.counter = 0
        self.display_level = logging.getLevelName(display_level.upper())

        # Validators
        self.experiment_file_validator = get_experiment_file_validator()
        self.ws_msg_validator = get_ws_msg_validator()
        self.os_packages_msg_validator = get_os_packages_msg_validator()
        self.graph_msg_validator = get_graph_msg_validator()
        self.system_details_msg_validator = get_system_details_msg_validator()
        self.cloud_details_msg_validator = get_cloud_details_msg_validator()
        self.upload_msg_validator = get_upload_msg_validator()
        self.remote_file_msg_validator = get_remote_file_msg_validator()

        self.server_address = get_backend_address()

        self._read_experiment_file()

        self.connection = RestServerConnection(
            self.api_key,
            self.experiment_id,
            self.server_address,
            self.config["comet.timeout.http"],
        )
        self.ws_connection = None
        self.rest_api_client = None
        self.focus_link = None

        self.file_upload_manager = FileUploadManager(
            self.config.get_int(None, "comet.internal.file_upload_worker_ratio")
        )
        self.file_upload_waiting_timeout = file_upload_waiting_timeout

        self.validation_error = validation_error

    def send(self):
        self._handshake()

        self._status_report_start()

        LOGGER.log(self.display_level, OFFLINE_SENDER_STARTS)

        self._send_messages()

        self._status_report_end()

        self._send_start_ends_time()

    def _read_experiment_file(self):
        with io.open(
            join(self.offline_dir, "experiment.json"), encoding="utf-8"
        ) as experiment_file:
            metadata = json.load(experiment_file)

        self.experiment_file_validator.validate(metadata)

        if self.force_reupload is True:
            self.experiment_id = generate_guid()
        else:
            self.experiment_id = metadata.get("offline_id")

            # Offline experiments created with old versions of the SDK will be
            # missing this field, so generate a new one if that's the case
            if not self.experiment_id:
                self.experiment_id = generate_guid()

        self.project_name = metadata["project_name"]
        self.start_time = metadata["start_time"]
        self.stop_time = metadata["stop_time"]
        self.tags = metadata["tags"]
        self.workspace = metadata["workspace"]
        self.mode = metadata.get("mode", "create")

    def _handshake(self):
        if self.mode == "create":
            run_id_results = self.connection.get_run_id(
                self.project_name, self.workspace, offline=True
            )
        elif self.mode == "append":
            run_id_results = self.connection.get_old_run_id(self.experiment_id)
        else:
            raise ValueError("Unknown mode value %r" % self.mode)

        (
            self.run_id,
            self.ws_url,
            self.project_id,
            self.is_github,
            self.focus_link,
            _,
            _,
            feature_toggles,
            initial_offset,
            upload_web_asset_url_prefix,
            upload_web_image_url_prefix,
            upload_api_asset_url_prefix,
            upload_api_image_url_prefix,
        ) = run_id_results

        # Send tags if present
        if self.tags:
            self.connection.add_tags(self.tags)

        full_ws_url = format_url(self.ws_url, apiKey=self.api_key, runId=self.run_id)

        self.ws_connection = WebSocketConnection(full_ws_url, self.connection)
        self.ws_connection.start()
        self.ws_connection.wait_for_connection()

        self.rest_api_client = get_rest_api_client("v2", api_key=self.api_key)

    def _send_messages(self):
        i = 0

        # Samples down the metrics
        sampling_size = self.config["comet.offline_sampling_size"]

        LOGGER.debug("Sampling metrics to %d values per metric name", sampling_size)

        sampler = MetricsSampler(sampling_size)

        with io.open(
            join(self.offline_dir, "messages.json"), encoding="utf-8"
        ) as messages_files:
            for i, line in enumerate(messages_files):
                try:
                    message = json.loads(line)

                    LOGGER.debug("Message %r", message)

                    message_type = message["type"]

                    if message_type == "ws_msg":
                        message_payload = message["payload"]
                        # Inject the offset now
                        message_payload["offset"] = i + 1
                        message_metric = message_payload.get("metric")

                        if message_metric:
                            sampler.sample_metric(message_payload)
                        else:
                            self._process_ws_msg(message_payload)
                    elif message_type == "file_upload":
                        self._process_upload_message(message)
                    elif message_type == "remote_file":
                        self._process_remote_file_message(message)
                    elif message_type == "os_packages":
                        self._process_os_packages_message(message)
                    elif message_type == "graph":
                        self._process_graph_message(message)
                    elif message_type == "system_details":
                        self._process_system_details_message(message)
                    elif message_type == "cloud_details":
                        self._process_cloud_details_message(message)
                    else:
                        raise ValueError("Unknown message type %r", message_type)
                except Exception:
                    LOGGER.warning("Error processing line %d", i + 1, exc_info=True)

        # Then send the sampled metrics
        samples = sampler.get_samples()
        for metric in samples:
            try:
                self._process_ws_msg(metric)
            except Exception:
                LOGGER.warning("Error processing metric", exc_info=True)

        LOGGER.debug("Done sending %d messages", i)

    def _process_ws_msg(self, message):
        try:
            self.ws_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_EXPERIMENT_INVALID_WS_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_WS_MSG, err_msg=tb)

        # Inject api key and run_id
        message["apiKey"] = self.api_key
        message["runId"] = self.run_id
        message["projectId"] = self.project_id
        message["experimentKey"] = self.experiment_id

        to_send = format_messages_for_ws([message])

        # The ws connection is created during handshake
        self.ws_connection.send(to_send)  # type: ignore

    def _process_upload_message(self, message):
        message = message["payload"]

        try:
            self.upload_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_UPLOAD_MSG, err_msg=tb)

        # Compute the url from the upload type
        url = self.connection.get_upload_url(message["upload_type"])

        additional_params = message["additional_params"] or {}
        additional_params["runId"] = self.run_id
        # Temporary fix to ensure integers:
        if "step" in additional_params and additional_params["step"] is not None:
            additional_params["step"] = int(additional_params["step"])
        if "epoch" in additional_params and additional_params["epoch"] is not None:
            additional_params["epoch"] = int(additional_params["epoch"])

        file_path = join(self.offline_dir, message["file_path"])
        file_size = os.path.getsize(file_path)

        self.file_upload_manager.upload_file_thread(
            project_id=self.project_id,
            experiment_id=self.experiment_id,
            file_path=file_path,
            upload_endpoint=url,
            api_key=self.api_key,
            additional_params=additional_params,
            clean=True,
            timeout=self.config.get_int(None, "comet.timeout.file_upload"),
            estimated_size=file_size,
        )
        LOGGER.debug("Processing uploading message done")

    def _process_remote_file_message(self, message):
        message = message["payload"]

        try:
            self.remote_file_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_UPLOAD_MSG, err_msg=tb)

        # Compute the url from the upload type
        url = self.connection.get_upload_url(message["upload_type"])

        additional_params = message["additional_params"] or {}
        additional_params["runId"] = self.run_id
        # Temporary fix to ensure integers:
        if "step" in additional_params and additional_params["step"] is not None:
            additional_params["step"] = int(additional_params["step"])
        if "epoch" in additional_params and additional_params["epoch"] is not None:
            additional_params["epoch"] = int(additional_params["epoch"])

        self.file_upload_manager.upload_remote_asset_thread(
            project_id=self.project_id,
            experiment_id=self.experiment_id,
            remote_uri=message["remote_uri"],
            upload_endpoint=url,
            api_key=self.api_key,
            additional_params=additional_params,
            metadata=message["metadata"],
            timeout=self.config.get_int(None, "comet.timeout.file_upload"),
        )
        LOGGER.debug("Processing remote uploading message done")

    def _process_os_packages_message(self, message):
        message = message["payload"]

        try:
            self.os_packages_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_INVALID_WS_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_WS_MSG, err_msg=tb)

        try:
            self.rest_api_client.set_experiment_os_packages(
                self.experiment_id, message["os_packages"]
            )
        except CometRestApiException as exc:
            LOGGER.error(
                OS_PACKAGE_MSG_SENDING_ERROR,
                exc.response.status_code,
                exc.response.content,
            )
        except Exception:
            LOGGER.error("Error sending os_packages message", exc_info=True)

    def _process_graph_message(self, message):
        message = message["payload"]

        try:
            self.graph_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_INVALID_WS_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_WS_MSG, err_msg=tb)

        try:
            self.rest_api_client.set_experiment_model_graph(
                self.experiment_id, message["graph"]
            )
        except CometRestApiException as exc:
            LOGGER.error(
                MODEL_GRAPH_MSG_SENDING_ERROR,
                exc.response.status_code,
                exc.response.content,
            )
        except Exception:
            LOGGER.error("Error sending os_packages message", exc_info=True)

    def _process_system_details_message(self, message):
        message = message["payload"]

        try:
            self.system_details_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_INVALID_WS_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_WS_MSG, err_msg=tb)

        try:
            self.rest_api_client.set_experiment_system_details(
                _os=message["os"],
                command=message["command"],
                env=message["env"],
                experiment_key=self.experiment_id,
                hostname=message["hostname"],
                ip=message["ip"],
                machine=message["machine"],
                os_release=message["os_release"],
                os_type=message["os_type"],
                pid=message["pid"],
                processor=message["processor"],
                python_exe=message["python_exe"],
                python_version_verbose=message["python_version_verbose"],
                python_version=message["python_version"],
                user=message["user"],
            )
        except CometRestApiException as exc:
            LOGGER.error(
                MODEL_GRAPH_MSG_SENDING_ERROR,
                exc.response.status_code,
                exc.response.content,
            )
        except Exception:
            LOGGER.error("Error sending os_packages message", exc_info=True)

    def _process_cloud_details_message(self, message):
        message = message["payload"]

        try:
            self.cloud_details_msg_validator.validate(message)
        except ValidationError:
            if self.validation_error:
                raise

            tb = traceback.format_exc()
            LOGGER.warning(OFFLINE_INVALID_WS_MSG, exc_info=True)
            self.connection.report(event_name=OFFLINE_INVALID_WS_MSG, err_msg=tb)

        try:
            self.rest_api_client.set_experiment_cloud_details(
                experiment_key=self.experiment_id,
                provider=message["provider"],
                cloud_metadata=message["cloud_metadata"],
            )
        except CometRestApiException as exc:
            LOGGER.error(
                CLOUD_DETAILS_MSG_SENDING_ERROR,
                exc.response.status_code,
                exc.response.content,
            )
        except Exception:
            LOGGER.error("Error sending cloud details message", exc_info=True)

    def _status_report_start(self):
        self.connection.update_experiment_status(
            self.run_id, self.project_id, True, offline=True
        )

    def _status_report_end(self):
        self.connection.update_experiment_status(
            self.run_id, self.project_id, False, offline=True
        )

    def _send_start_ends_time(self):
        if self.mode == "create":
            self.connection.offline_experiment_start_end_time(
                self.run_id, self.start_time, self.stop_time
            )
        elif self.mode == "append":
            self.connection.offline_experiment_start_end_time(
                self.run_id, None, self.stop_time
            )
        else:
            # This shouldn't happen in the normal flow as the invalid mode is validated during
            # handshake
            raise ValueError("Invalid offline mode %r" % self.mode)

    def _get_experiment_url(self):
        if self.focus_link:
            return self.focus_link + self.experiment_id

        return ""

    def close(self):
        LOGGER.info(WAITING_DATA_UPLOADED)
        if self.ws_connection is not None:
            self.ws_connection.close()
            # Use 3 times the default timeout as they might be a lot of backend messages to process
            ws_cleaned = self.ws_connection.wait_for_finish(
                timeout=DEFAULT_WS_JOIN_TIMEOUT * 3
            )

            if not ws_cleaned:
                LOGGER.error(
                    "Failed to send all messages, metrics and output will likely be incomplete"
                )

        if self.rest_api_client is not None:
            self.rest_api_client.close()

        self.file_upload_manager.close()
        # Finish remained uploads and display upload progress
        if not self.file_upload_manager.all_done():
            monitor = FileUploadManagerMonitor(self.file_upload_manager)
            LOGGER.info(FILE_UPLOADS_PROMPT)

            wait_for_done(
                monitor.all_done,
                self.file_upload_waiting_timeout,
                progress_callback=monitor.log_remaining_uploads,
                sleep_time=15,
            )

        self.file_upload_manager.join()
        LOGGER.debug("Upload threads %r", self.file_upload_manager)

        LOGGER.log(self.display_level, OFFLINE_SENDER_ENDS, self._get_experiment_url())


def upload_single_offline_experiment(
    offline_archive_path, api_key, force_reupload, display_level="info"
):
    unzipped_directory = unzip_offline_archive(offline_archive_path)
    sender = OfflineSender(
        api_key,
        unzipped_directory,
        force_reupload=force_reupload,
        display_level=display_level,
    )
    try:
        sender.send()
        sender.close()
        return True
    except ExperimentAlreadyUploaded:
        LOGGER.error(OFFLINE_EXPERIMENT_ALREADY_UPLOADED, offline_archive_path)
        return False
    finally:
        try:
            shutil.rmtree(unzipped_directory)
        except OSError:
            # We made our best effort to clean after ourselves
            msg = "Failed to clean the Offline sender tmpdir %r"
            LOGGER.debug(msg, unzipped_directory, exc_info=True)


def main_upload(archives, force_reupload):
    upload_count = 0
    fail_count = 0

    # Common code
    config = get_config()
    api_key = get_api_key(None, config)

    for filename in archives:
        LOGGER.info("Attempting to upload '%s'...", filename)
        try:
            success = upload_single_offline_experiment(
                filename, api_key, force_reupload
            )

            if success:
                upload_count += 1
            else:
                fail_count += 1

        except KeyboardInterrupt:
            break
        except InvalidAPIKey:
            LOGGER.error(
                "comet upload failed because of invalid api key; please set COMET_API_KEY"
            )
        except Exception:
            LOGGER.error(
                "    Upload failed", exc_info=True, extra={"show_traceback": True}
            )
            fail_count += 1
        else:
            LOGGER.info("    done!")
    LOGGER.info("Number of uploaded experiments: %s", upload_count)
    if fail_count > 0:
        LOGGER.info("Number of failed experiment uploads: %s", fail_count)


def _create_zip_file(directory, name):
    file_path = _offline_zip_path(directory, name)
    return ZipFile(file_path, "w", allowZip64=True)


def _offline_zip_path(directory, name):
    return os.path.join(directory, "%s.zip" % name)
