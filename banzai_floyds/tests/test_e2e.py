import pytest
import time
import logging
from banzai.celery import app
from banzai.utils import file_utils

logger = logging.getLogger('banzai')

app.conf.update(CELERY_TASK_ALWAYS_EAGER=True)

DATA_FILELIST = pkg_resources.resource_filename(TEST_PACKAGE, 'data/configdb_example.json')


def celery_join():
    celery_inspector = app.control.inspect()
    log_counter = 0
    while True:
        queues = [celery_inspector.active(), celery_inspector.scheduled(), celery_inspector.reserved()]
        time.sleep(1)
        log_counter += 1
        if log_counter % 30 == 0:
            logger.info('Processing: ' + '. ' * (log_counter // 30))
        queue_names = []
        for queue in queues:
            if queue is not None:
                queue_names += queue.keys()
        if 'celery@banzai-celery-worker' not in queue_names:
            logger.warning('No valid celery queues were detected, retrying...', extra_tags={'queues': queues})
            # Reset the celery connection
            celery_inspector = app.control.inspect()
            continue
        if all(queue is None or len(queue['celery@banzai-celery-worker']) == 0 for queue in queues):
            break


# Note this is complicated by the fact that things are running as celery tasks.
@pytest.mark.e2e
@pytest.fixture(scope='module')
@mock.patch('banzai.main.argparse.ArgumentParser.parse_args')
@mock.patch('banzai.main.file_utils.post_to_ingester', return_value={'frameid': None})
@mock.patch('banzai.dbs.requests.get', return_value=FakeResponse(CONFIGDB_FILENAME))
def init(configdb, mock_ingester, mock_args):
    os.system(f'banzai_create_db --db-address={os.environ["DB_ADDRESS"]}')
    populate_instrument_tables(db_address=os.environ["DB_ADDRESS"], configdb_address='http://fakeconfigdb')
    for instrument in INSTRUMENTS:
        for bpm_filepath in glob(os.path.join(DATA_ROOT, instrument, 'bpm/*bpm*')):
            mock_args.return_value = argparse.Namespace(filepath=bpm_filepath, db_address=os.environ['DB_ADDRESS'],
                                                        log_level='debug')
            add_bpm()

@pytest.mark.e2e
@pytest.mark.science_frames
class TestScienceFileCreation:
    @pytest.fixture(autouse=True)
    def process_science_frames(self, init):
        logger.info('Reducing individual frames for filenames: {filenames}'.format(filenames=raw_filenames))
        for day_obs in DAYS_OBS:
            raw_path = os.path.join(DATA_ROOT, day_obs, 'raw')
            for filename in glob(os.path.join(raw_path, raw_filenames)):
                file_utils.post_to_archive_queue(filename, os.getenv('FITS_BROKER'),
                                                 exchange_name=os.getenv('FITS_EXCHANGE'))
        celery_join()
        logger.info('Finished reducing individual frames for filenames: {filenames}'.format(filenames=raw_filenames))

    def test_if_science_frames_were_created(self):
        expected_files = []
        created_files = []
        for day_obs in DAYS_OBS:
            expected_files += [os.path.basename(filename).replace('e00', 'e91')
                               for filename in glob(os.path.join(DATA_ROOT, day_obs, 'raw', '*e00*'))]
            created_files += [os.path.basename(filename) for filename in glob(os.path.join(DATA_ROOT, day_obs,
                                                                                           'processed', '*e91*'))]
        assert len(expected_files) > 0
        for expected_file in expected_files:
            assert expected_file in created_files