
from __future__ import annotations
import json
import logging
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import os
import random

from tqdm import tqdm

logger = logging.getLogger('ParallelRunner')


def chunk(li: list, chunk_size: int) -> list[list]:
    return [li[i:i+chunk_size] for i in range(0, len(li), chunk_size)]


class ParallelRunner:
    def __init__(self, 
                key: str,
                num_workers=2,
                chunk_size=1,
                verbose=True) -> None:
        self.key = key
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.query_results = []
        self._query_func = None
        self.verbose = verbose
        self._file = None


    @staticmethod
    def query_once(item, query_func: callable, key: str, debug=False):
        result = None
        if debug:
            result = query_func(item, key)
        else:
            try:
                result = query_func(item, key)
            except Exception as e:
                logger.error(str(e))
                return None
        return result


    @staticmethod
    def query(data, query_func, key):
        """The entrance function executed by worker process(es)."""
        results = []
        for item in data:
            while True:
                result = ParallelRunner.query_once(item, query_func, key)
                # Got result
                if result is not None:
                    results.append(result)
                    break
        return results
    

    @staticmethod
    def query_batch(data, query_func, key):
        """The entrance function executed by worker process(es)."""
        results = None
        while True:
            results = ParallelRunner.query_once(data, query_func, key)
            if results is not None and isinstance(results, list):
                break
        return results
    

    def handle_result(self, results):
        if results:
            for item in results:
                self._file.write(json.dumps(item))
                self._file.write('\n')
                self._file.flush()
            self.query_results.extend(results)
        if self.num_workers > 0 and self.verbose:
            self.progress_bar.update(self.chunk_size)


    def handle_error(self, error):
        print(error, flush=True)


    def filter_unfinished_jobs(self, data, output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            saved_jobs = (json.loads(js_str) for js_str in f.read().strip().split('\n'))
            saved_job_ids = set([job['job_id'] for job in saved_jobs])
        print(f"Found {len(saved_job_ids)} saved jobs.")
        return list(filter(lambda item: item['job_id'] not in saved_job_ids, data))


    def start(self, data, query_func, output_filename, resume=False, batch=False):
        if resume:
            data = self.filter_unfinished_jobs(data, output_filename)

        self._file = open(output_filename, 'a', encoding='utf-8')

        if self.verbose:
            print("Starting jobs...")
        if self.num_workers == 0:
            iterator = tqdm(data) if self.verbose else data
            for item in iterator:
                results = self.query((item,), query_func, self.keys, self.cfg)
                self.handle_result(results)
        else:
            pool = ThreadPool(self.num_workers)
            self._query_func = query_func
            self.progress_bar = tqdm(total=len(data)) if self.verbose else None
            data_chunks = chunk(data, self.chunk_size)
            for dt_i, data in enumerate(data_chunks):
                if batch:
                    pool.apply_async(self.query_batch, 
                                     (data, query_func, self.key), 
                                     callback=self.handle_result, error_callback=self.handle_error)
                else:
                    pool.apply_async(self.query, 
                                     (data, query_func, self.key), 
                                     callback=self.handle_result, error_callback=self.handle_error)
            pool.close()
            pool.join()
            if self.verbose:
                self.progress_bar.close()
        if self.verbose:
            print("Jobs finished.")
        return self.query_results
