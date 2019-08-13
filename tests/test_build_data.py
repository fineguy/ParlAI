#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core import build_data
import unittest
import parlai.core.testing_utils as testing_utils
from parlai.core.params import ParlaiParser
import requests_mock


class TestBuildData(unittest.TestCase):
    """
    Basic tests on the build_data.py download_multiprocess.
    """

    dest_filenames = ['mnist0.tar.gz', 'mnist1.tar.gz', 'mnist2.tar.gz']

    def setUp(self):
        self.datapath = ParlaiParser().parse_args(print_args=False)['datapath']
        for d in self.dest_filenames:
            # Removing files if they are already there b/c otherwise it won't try to download them again
            try:
                os.remove(os.path.join(self.datapath, d))
            except Exception:
                pass

    def test_download_multiprocess(self):
        with requests_mock.Mocker() as requests_mocker:
            # Mocking the requests b/c we get an inexplicable 408 (timeout)
            # on good files only when running on CircleCI (not locally)
            requests_mocker.get('http://parl.ai/downloads/mnist/mnist.tar.gz', text='data')
            requests_mocker.get('http://parl.ai/downloads/mnist/mnist.tar.gz.BAD', text='data', status_code=403)

            with testing_utils.capture_output() as stdout:
                urls = [
                    'http://parl.ai/downloads/mnist/mnist.tar.gz',
                    'http://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
                    'http://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
                ]

                download_results = build_data.download_multiprocess(
                    urls, self.datapath, dest_filenames=self.dest_filenames
                )

            str_output = stdout.getvalue()
            # output_filenames, output_statuses = zip(*download_results)
            print('stdout output: %s' % str_output)
            output_filenames = [
                download_results[0][0],
                download_results[1][0],
                download_results[2][0],
            ]
            output_statuses = [
                download_results[0][1],
                download_results[1][1],
                download_results[2][1],
            ]
            self.assertEqual(
                output_filenames,
                ['mnist0.tar.gz', 'mnist1.tar.gz', 'mnist2.tar.gz'],
                'output filenames not correct',
            )
            self.assertEqual(
                output_statuses, [200, 403, 403], 'output http statuses not correct'
            )

    def test_download_multiprocess_chunks(self):
        # Tests that the three finish downloading but may finish in any order
        with requests_mock.Mocker() as requests_mocker:
            # Mocking the requests b/c we get an inexplicable 408 (timeout)
            # on good files only when running on CircleCI (not locally)
            requests_mocker.get('http://parl.ai/downloads/mnist/mnist.tar.gz', text='data')
            requests_mocker.get('http://parl.ai/downloads/mnist/mnist.tar.gz.BAD', text='data', status_code=403)
            with testing_utils.capture_output() as stdout:
                urls = [
                    'http://parl.ai/downloads/mnist/mnist.tar.gz',
                    'http://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
                    'http://parl.ai/downloads/mnist/mnist.tar.gz.BAD',
                ]

                download_results = build_data.download_multiprocess(
                    urls, self.datapath, dest_filenames=self.dest_filenames, chunk_size=1
                )

            str_output = stdout.getvalue()
            print('stdout output: %s' % str_output)
            output_filenames = [
                download_results[0][0],
                download_results[1][0],
                download_results[2][0],
            ]
            output_statuses = [
                download_results[0][1],
                download_results[1][1],
                download_results[2][1],
            ]
            self.assertIn('mnist0.tar.gz', output_filenames)
            self.assertIn('mnist1.tar.gz', output_filenames)
            self.assertIn('mnist2.tar.gz', output_filenames)
            self.assertIn(200, output_statuses)
            self.assertIn(403, output_statuses)


if __name__ == '__main__':
    unittest.main()
