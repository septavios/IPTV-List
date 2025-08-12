#!/usr/bin/env python3
"""
Enhanced IPTV Stream Connectivity Checker
Optimized for large playlists with advanced multithreading and memory management
"""

import requests
import re
import logging
import concurrent.futures
import threading
import time
import gc
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse
import pandas as pd
from typing import List, Dict, Optional, Tuple
import queue
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('iptv_checker.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedIPTVChecker:
    """Enhanced IPTV Stream Checker with optimizations for large playlists"""
    
    def __init__(self, m3u8_url: str, timeout: int = 10, max_workers: int = None, 
                 batch_size: int = 1000, memory_limit_mb: int = 2048):
        """
        Initialize the enhanced IPTV checker
        
        Args:
            m3u8_url: URL of the M3U8 playlist
            timeout: Request timeout in seconds
            max_workers: Maximum number of worker threads (auto-calculated if None)
            batch_size: Number of channels to process in each batch
            memory_limit_mb: Memory limit in MB before triggering garbage collection
        """
        self.m3u8_url = m3u8_url
        self.timeout = timeout
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        
        # Auto-calculate optimal worker count based on system resources
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            # Use 2x CPU cores for I/O bound tasks, but cap at 100 for very large systems
            self.max_workers = min(cpu_count * 2, 100)
        else:
            self.max_workers = max_workers
            
        self.channels = []
        self.results = []
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.start_time = None
        
        # Session pool for connection reuse
        self.session_pool = queue.Queue()
        self._init_session_pool()
        
        logger.info(f"Enhanced IPTV Checker initialized:")
        logger.info(f"  - Max workers: {self.max_workers}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Timeout: {self.timeout}s")
        logger.info(f"  - Memory limit: {self.memory_limit_mb}MB")
    
    def _init_session_pool(self):
        """Initialize a pool of HTTP sessions for connection reuse"""
        for _ in range(self.max_workers):
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            # Configure session for better performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=1
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self.session_pool.put(session)
    
    def _get_session(self) -> requests.Session:
        """Get a session from the pool"""
        try:
            return self.session_pool.get_nowait()
        except queue.Empty:
            # Create new session if pool is empty
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            return session
    
    def _return_session(self, session: requests.Session):
        """Return a session to the pool"""
        try:
            self.session_pool.put_nowait(session)
        except queue.Full:
            # Close session if pool is full
            session.close()
    
    def _check_memory_usage(self):
        """Check memory usage and trigger garbage collection if needed"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.memory_limit_mb:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB). Running garbage collection...")
            gc.collect()
            new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage after GC: {new_memory_mb:.1f}MB")
    
    def download_m3u8(self) -> Optional[str]:
        """Download M3U8 playlist content with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading M3U8 playlist (attempt {attempt + 1}/{max_retries})...")
                session = self._get_session()
                
                response = session.get(self.m3u8_url, timeout=30)
                response.raise_for_status()
                
                content = response.text
                self._return_session(session)
                
                logger.info(f"Successfully downloaded M3U8 playlist ({len(content)} characters)")
                return content
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All download attempts failed")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def parse_m3u8(self, content: str) -> List[Dict]:
        """Parse M3U8 content and extract channel information"""
        logger.info("Parsing M3U8 content...")
        channels = []
        lines = content.strip().split('\n')
        
        current_channel = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#EXTINF:'):
                # Parse EXTINF line
                current_channel = {}
                
                # Extract attributes using regex
                attrs = re.findall(r'(\w+)="([^"]*)"', line)
                for attr, value in attrs:
                    current_channel[attr.lower()] = value
                
                # Extract channel name (after the last comma)
                if ',' in line:
                    name_part = line.split(',', 1)[1].strip()
                    current_channel['name'] = name_part
                
            elif line and not line.startswith('#') and current_channel:
                # This is a URL line
                current_channel['url'] = line
                channels.append(current_channel.copy())
                current_channel = {}
        
        logger.info(f"Parsed {len(channels)} channels from M3U8 playlist")
        return channels
    
    def test_stream(self, channel: Dict) -> Dict:
        """Test a single stream with enhanced error handling"""
        channel_name = channel.get('name', 'Unknown')
        url = channel.get('url', '')
        
        if not url:
            return {
                'channel': channel,
                'status': 'No URL',
                'response_time': 0,
                'error': 'No URL provided'
            }
        
        session = self._get_session()
        start_time = time.time()
        
        try:
            # Use HEAD request first for faster checking
            response = session.head(url, timeout=self.timeout, allow_redirects=True)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                status = 'Working'
                error = None
            elif response.status_code == 405:  # Method not allowed, try GET
                response = session.get(url, timeout=self.timeout, stream=True)
                response_time = time.time() - start_time
                if response.status_code == 200:
                    status = 'Working'
                    error = None
                else:
                    status = 'Failed'
                    error = f"HTTP {response.status_code}"
            else:
                status = 'Failed'
                error = f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            response_time = self.timeout
            status = 'Timeout'
            error = f"Timeout after {self.timeout}s"
        except requests.exceptions.ConnectionError:
            response_time = time.time() - start_time
            status = 'Connection Error'
            error = "Connection failed"
        except Exception as e:
            response_time = time.time() - start_time
            status = 'Error'
            error = str(e)
        finally:
            self._return_session(session)
        
        result = {
            'channel': channel,
            'status': status,
            'response_time': round(response_time, 2),
            'error': error
        }
        
        # Update progress with thread safety
        with self.progress_lock:
            self.processed_count += 1
            if self.processed_count % 50 == 0 or self.processed_count % 10 == 0 and self.processed_count <= 100:
                elapsed = time.time() - self.start_time
                rate = self.processed_count / elapsed
                eta = (len(self.channels) - self.processed_count) / rate if rate > 0 else 0
                
                logger.info(f"Progress: {self.processed_count}/{len(self.channels)} "
                          f"({self.processed_count/len(self.channels)*100:.1f}%) | "
                          f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")
        
        # Log individual results less frequently for large playlists
        if len(self.channels) < 1000 or self.processed_count % 100 == 0:
            with self.lock:
                logger.info(f"Tested: {channel_name[:30]:<30} | {status:<15} | {response_time:.2f}s")
        
        return result
    
    def test_batch(self, batch_channels: List[Dict]) -> List[Dict]:
        """Test a batch of channels"""
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_channel = {executor.submit(self.test_stream, channel): channel 
                               for channel in batch_channels}
            
            for future in concurrent.futures.as_completed(future_to_channel):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing channel: {e}")
        
        return batch_results
    
    def test_all_streams(self):
        """Test all streams with batch processing for memory efficiency"""
        logger.info(f"Testing {len(self.channels)} streams in batches of {self.batch_size}...")
        self.start_time = time.time()
        self.processed_count = 0
        
        # Process channels in batches to manage memory
        for i in range(0, len(self.channels), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.channels))
            batch_channels = self.channels[i:batch_end]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(self.channels)-1)//self.batch_size + 1} "
                       f"(channels {i+1}-{batch_end})")
            
            batch_results = self.test_batch(batch_channels)
            self.results.extend(batch_results)
            
            # Check memory usage and clean up if needed
            self._check_memory_usage()
            
            # Small delay between batches to prevent overwhelming the system
            if i + self.batch_size < len(self.channels):
                time.sleep(0.1)
        
        total_time = time.time() - self.start_time
        logger.info(f"Completed testing {len(self.channels)} streams in {total_time:.1f}s "
                   f"(avg: {total_time/len(self.channels):.2f}s per stream)")
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive summary statistics"""
        if not self.results:
            return {}
        
        total_channels = len(self.results)
        working_channels = sum(1 for r in self.results if r['status'] == 'Working')
        failed_channels = sum(1 for r in self.results if r['status'] == 'Failed')
        timeout_channels = sum(1 for r in self.results if r['status'] == 'Timeout')
        error_channels = sum(1 for r in self.results if r['status'] in ['Connection Error', 'Error', 'No URL'])
        
        # Group by category
        groups = defaultdict(lambda: {'total': 0, 'working': 0, 'failed': 0, 'timeout': 0, 'error': 0})
        for result in self.results:
            group = result['channel'].get('group', 'Unknown')
            groups[group]['total'] += 1
            status = result['status']
            if status == 'Working':
                groups[group]['working'] += 1
            elif status == 'Failed':
                groups[group]['failed'] += 1
            elif status == 'Timeout':
                groups[group]['timeout'] += 1
            else:
                groups[group]['error'] += 1
        
        # Response time statistics for working channels
        working_times = [r['response_time'] for r in self.results if r['status'] == 'Working']
        if working_times:
            avg_response_time = sum(working_times) / len(working_times)
            min_response_time = min(working_times)
            max_response_time = max(working_times)
            median_response_time = sorted(working_times)[len(working_times)//2]
        else:
            avg_response_time = min_response_time = max_response_time = median_response_time = 0
        
        summary = {
            'total_channels': total_channels,
            'working_channels': working_channels,
            'failed_channels': failed_channels,
            'timeout_channels': timeout_channels,
            'error_channels': error_channels,
            'success_rate': round((working_channels / total_channels) * 100, 2) if total_channels > 0 else 0,
            'avg_response_time': round(avg_response_time, 2),
            'min_response_time': round(min_response_time, 2),
            'max_response_time': round(max_response_time, 2),
            'median_response_time': round(median_response_time, 2),
            'groups': dict(groups),
            'test_duration': getattr(self, 'start_time', 0) and time.time() - self.start_time
        }
        
        return summary
    
    def save_detailed_report(self, filename: str = None) -> bool:
        """Save detailed report to Excel with memory-efficient processing"""
        if not self.results:
            logger.warning("No results to save")
            return False
        
        if filename is None:
            filename = f"iptv_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            logger.info(f"Saving detailed report to {filename}...")
            
            # Process results in chunks to manage memory for large datasets
            chunk_size = 10000
            all_data = []
            
            for i in range(0, len(self.results), chunk_size):
                chunk_results = self.results[i:i + chunk_size]
                chunk_data = []
                
                for result in chunk_results:
                    channel = result['channel']
                    chunk_data.append({
                        'Channel Name': channel.get('name', ''),
                        'Group': channel.get('group', ''),
                        'TVG ID': channel.get('tvg_id', ''),
                        'TVG Name': channel.get('tvg_name', ''),
                        'Stream URL': channel.get('url', ''),
                        'Status': result['status'],
                        'Response Time (s)': result['response_time'],
                        'Error': result['error'] or '',
                        'Logo URL': channel.get('logo', '')
                    })
                
                all_data.extend(chunk_data)
                
                # Check memory usage
                if i > 0 and i % (chunk_size * 5) == 0:
                    self._check_memory_usage()
            
            df = pd.DataFrame(all_data)
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # All results
                df.to_excel(writer, sheet_name='All Channels', index=False)
                
                # Working channels only
                working_df = df[df['Status'] == 'Working'].copy()
                if not working_df.empty:
                    working_df = working_df.sort_values('Response Time (s)')
                    working_df.to_excel(writer, sheet_name='Working Channels', index=False)
                
                # Failed channels only
                failed_df = df[df['Status'] != 'Working'].copy()
                if not failed_df.empty:
                    failed_df.to_excel(writer, sheet_name='Failed Channels', index=False)
                
                # Summary by group
                summary_data = []
                summary = self.generate_summary()
                for group, stats in summary['groups'].items():
                    summary_data.append({
                        'Group': group,
                        'Total Channels': stats['total'],
                        'Working Channels': stats['working'],
                        'Failed Channels': stats['failed'],
                        'Timeout Channels': stats['timeout'],
                        'Error Channels': stats['error'],
                        'Success Rate (%)': round((stats['working'] / stats['total']) * 100, 2) if stats['total'] > 0 else 0
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Success Rate (%)', ascending=False)
                summary_df.to_excel(writer, sheet_name='Summary by Group', index=False)
                
                # Performance statistics
                perf_data = [{
                    'Metric': 'Total Channels',
                    'Value': summary['total_channels']
                }, {
                    'Metric': 'Working Channels',
                    'Value': summary['working_channels']
                }, {
                    'Metric': 'Success Rate (%)',
                    'Value': summary['success_rate']
                }, {
                    'Metric': 'Average Response Time (s)',
                    'Value': summary['avg_response_time']
                }, {
                    'Metric': 'Minimum Response Time (s)',
                    'Value': summary['min_response_time']
                }, {
                    'Metric': 'Maximum Response Time (s)',
                    'Value': summary['max_response_time']
                }, {
                    'Metric': 'Median Response Time (s)',
                    'Value': summary['median_response_time']
                }, {
                    'Metric': 'Test Duration (s)',
                    'Value': round(summary.get('test_duration', 0), 1)
                }]
                
                perf_df = pd.DataFrame(perf_data)
                perf_df.to_excel(writer, sheet_name='Performance Stats', index=False)
            
            logger.info(f"Detailed report saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return False
    
    def print_summary_report(self):
        """Print comprehensive summary report to console"""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("ENHANCED IPTV STREAM CONNECTIVITY REPORT")
        print("="*80)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Source: {self.m3u8_url}")
        print(f"Configuration: {self.max_workers} workers, {self.timeout}s timeout, {self.batch_size} batch size")
        print("-"*80)
        
        print(f"Total Channels Tested: {summary['total_channels']:,}")
        print(f"Working Channels: {summary['working_channels']:,} ({summary['success_rate']}%)")
        print(f"Failed Channels: {summary['failed_channels']:,}")
        print(f"Timeout Channels: {summary['timeout_channels']:,}")
        print(f"Error Channels: {summary['error_channels']:,}")
        
        if summary.get('test_duration'):
            print(f"Test Duration: {summary['test_duration']:.1f}s")
            print(f"Average Rate: {summary['total_channels']/summary['test_duration']:.1f} channels/second")
        
        print(f"\nResponse Time Statistics (Working Channels):")
        print(f"  Average: {summary['avg_response_time']}s")
        print(f"  Minimum: {summary['min_response_time']}s")
        print(f"  Maximum: {summary['max_response_time']}s")
        print(f"  Median: {summary['median_response_time']}s")
        
        print("\n" + "-"*80)
        print("RESULTS BY GROUP:")
        print("-"*80)
        print(f"{'Group':<30} | {'Working':<10} | {'Total':<8} | {'Success %':<10}")
        print("-"*80)
        
        # Sort groups by success rate
        sorted_groups = sorted(summary['groups'].items(), 
                             key=lambda x: (x[1]['working'] / x[1]['total']) if x[1]['total'] > 0 else 0, 
                             reverse=True)
        
        for group, stats in sorted_groups:
            success_rate = round((stats['working'] / stats['total']) * 100, 2) if stats['total'] > 0 else 0
            print(f"{group[:29]:<30} | {stats['working']:>9,} | {stats['total']:>7,} | {success_rate:>9.1f}%")
        
        print("\n" + "-"*80)
        print("TOP 15 FASTEST WORKING CHANNELS:")
        print("-"*80)
        
        working_results = [r for r in self.results if r['status'] == 'Working']
        working_results.sort(key=lambda x: x['response_time'])
        
        for i, result in enumerate(working_results[:15]):
            channel_name = result['channel'].get('name', 'Unknown')[:50]
            group = result['channel'].get('group', 'Unknown')[:15]
            response_time = result['response_time']
            print(f"{i+1:2d}. {channel_name:<50} | {group:<15} | {response_time:>6.2f}s")
        
        if len(working_results) == 0:
            print("No working channels found!")
        
        print("="*80)
    
    def run(self) -> bool:
        """Main execution method with enhanced error handling"""
        try:
            # Download M3U8 playlist
            content = self.download_m3u8()
            if not content:
                logger.error("Failed to download M3U8 playlist")
                return False
            
            # Parse channels
            self.channels = self.parse_m3u8(content)
            if not self.channels:
                logger.error("No channels found in M3U8 playlist")
                return False
            
            # Test all streams
            self.test_all_streams()
            
            # Generate and display report
            self.print_summary_report()
            
            # Save detailed report
            output_file = f"iptv_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            if self.save_detailed_report(output_file):
                print(f"\n✅ Detailed report saved to: {output_file}")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("Test interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")
            return False
        finally:
            # Clean up session pool
            while not self.session_pool.empty():
                try:
                    session = self.session_pool.get_nowait()
                    session.close()
                except queue.Empty:
                    break

def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced IPTV Stream Connectivity Checker')
    parser.add_argument('--url', default="https://iptv-org.github.io/iptv/index.m3u", 
                       help='M3U8 playlist URL')
    parser.add_argument('--timeout', type=int, default=10, 
                       help='Request timeout in seconds')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker threads (auto-calculated if not specified)')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Batch size for processing channels')
    parser.add_argument('--memory-limit', type=int, default=2048, 
                       help='Memory limit in MB')
    
    args = parser.parse_args()
    
    print("Enhanced IPTV Stream Connectivity Checker")
    print("="*50)
    
    # Create checker instance
    checker = EnhancedIPTVChecker(
        m3u8_url=args.url,
        timeout=args.timeout,
        max_workers=args.workers,
        batch_size=args.batch_size,
        memory_limit_mb=args.memory_limit
    )
    
    # Run the test
    success = checker.run()
    
    if not success:
        print("❌ IPTV testing failed")
        return 1
    
    print("✅ IPTV testing completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())