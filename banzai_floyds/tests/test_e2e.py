import pytest
import time
import logging
from banzai.celery import app
from banzai.tests.utils import FakeResponse
import banzai.dbs
import os
import pkg_resources
from kombu import Connection, Exchange
import mock
import requests
from astropy.io import fits, ascii
import numpy as np
from banzai.utils.fits_utils import download_from_s3
import banzai.main
from banzai_floyds import settings
from banzai.utils import file_utils


logger = logging.getLogger('banzai')

app.conf.update(CELERY_TASK_ALWAYS_EAGER=True)

DATA_FILELIST = pkg_resources.resource_filename('banzai_floyds.tests', 'data/test_data.dat')
CONFIGDB_FILENAME = pkg_resources.resource_filename('banzai_floyds.tests', 'data/configdb.json')


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
@mock.patch('banzai.dbs.requests.get', return_value=FakeResponse(CONFIGDB_FILENAME))
def init(mock_configdb):
    banzai.dbs.create_db(os.environ["DB_ADDRESS"])
    banzai.dbs.populate_instrument_tables(db_address=os.environ["DB_ADDRESS"], configdb_address='http://fakeconfigdb')


@pytest.mark.e2e
@pytest.mark.detect_orders
class TestOrderDetection:
    @pytest.fixture(autouse=True)
    def process_skyflat(self, init):
        # Pull down our experimental skyflat
        skyflat_info = ascii.read(pkg_resources.resource_filename('banzai_floyds.tests', 'data/test_skyflat.dat'))[0]
        skyflat_info = dict(skyflat_info)
        context = banzai.main.parse_args(settings, parse_system_args=False)
        skyflat_hdu = fits.open(download_from_s3(skyflat_info, context))

        # Munge the data to be OBSTYPE SKYFLAT
        skyflat_hdu['SCI'].header['OBSTYPE'] = 'SKYFLAT'
        skyflat_name = skyflat_info["filename"].replace("x00.fits", "f00.fits")
        skyflat_hdu.writeto(f'{skyflat_name}')
        # Process the data
        filename = os.path.join(os.getcwd(), skyflat_name)
        file_utils.post_to_archive_queue(filename, os.getenv('FITS_BROKER'),
                                         exchange_name=os.getenv('FITS_EXCHANGE'))

        celery_join()

    def test_that_order_mask_exists(self):
        expected_file = os.path.join()
        assert os.path.exists(expected_file)
        hdu = fits.open(expected_file)
        assert 'ORDERS' in hdu
        # Note there are only two orders in floyds
        assert np.max(hdu['ORDERS'].data) == 2


@pytest.mark.e2e
@pytest.mark.science_frames
class TestScienceFileCreation:
    def process_science_frames(self):
        logger.info('Reducing individual frames')

        exchange = Exchange(os.getenv('FITS_EXCHANGE', 'fits_files'), type='fanout')
        test_data = ascii.read(DATA_FILELIST)
        with Connection(os.getenv('FITS_BROKER')) as conn:
            producer = conn.Producer(exchange=exchange)
            for row in test_data:
                archive_record = requests.get(f'{os.getenv("API_ROOT")}frames/{row["frameid"]}').json()
                archive_record['frameid'] = archive_record['id']
                producer.publish(archive_record)
            producer.release()

        celery_join()
        logger.info('Finished reducing individual frames')

    def test_if_science_frames_were_created(self):
        test_data = ascii.read(DATA_FILELIST)
        for row in test_data:
            site = row['filename'][:3]
            camera = row['filename'].split('-')[1]
            dayobs = row['filename'].split('-')[2]
            expected_file = os.path.join('/archive', 'engineering', site, camera, dayobs, 'processed',
                                         row['filename'].replace('00.fits', '91.fits'))
            assert os.path.exists(expected_file)
