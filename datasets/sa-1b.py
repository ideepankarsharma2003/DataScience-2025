import asyncio
import os

# Directories and logs
SAVE_DIR = "SA-1B"
FAILED_LOG = "async-failed.txt"
os.makedirs(SAVE_DIR, exist_ok=True)

# Read URLs from file
with open("sa-1b.txt", "r") as file:
    lines = file.readlines()[-1:0:-1]  # Skip header line

# Parse file names and URLs
files_to_download = [line.strip().split("\t") for line in lines]

# Limit concurrency to avoid network overload
SEM = asyncio.Semaphore(10)  # Adjust based on system/network capacity


async def log_failed_download(file_name, url):
    """Log failed downloads asynchronously to a file."""
    async with asyncio.Lock():
        with open(FAILED_LOG, "a") as f:
            f.write(f"{file_name}\t{url}\n")


async def download_file(file_name, url, max_retries=5):
    """Asynchronously download a file using curl with retries."""
    file_path = os.path.join(SAVE_DIR, file_name)

    

    async with SEM:
        for attempt in range(1, max_retries + 1):
            print(f"[Attempt {attempt}] Downloading {file_name}...")

            process = await asyncio.create_subprocess_exec(
                "curl", "-L", "-C", "-", "-o", file_path, "--retry", "5", "--retry-delay", "10",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"✅ Successfully downloaded: {file_name}")
                return  # Exit function if download succeeds
            else:
                print(f"❌ Failed attempt {attempt} for {file_name}: {stderr.decode().strip()}")

            await asyncio.sleep(10)  # Wait before retrying

        print(f"🚨 Failed to download {file_name} after {max_retries} attempts.")
        await log_failed_download(file_name, url)


async def main():
    """Main function to manage async downloads."""
    tasks = [download_file(name, url) for name, url in files_to_download]
    await asyncio.gather(*tasks)

# Run the async downloader
asyncio.run(main())

print("\n✅ Download process completed. Check 'failed.txt' for any failed downloads.")
