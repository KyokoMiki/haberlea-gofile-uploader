"""Gofile upload extension for Haberlea.

This extension collects all downloaded directories and uploads them
as a single ZIP archive to Gofile.io after all downloads complete.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import aiohttp
import msgspec
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
)

from haberlea.plugins.base import ExtensionBase
from haberlea.utils.models import ExtensionInformation
from haberlea.utils.progress import BinaryTransferSpeedColumn

if TYPE_CHECKING:
    from haberlea.download_queue import DownloadJob
from haberlea.utils.utils import compress_to_zip, delete_path

logger = logging.getLogger(__name__)

# Regional upload endpoints
UPLOAD_REGIONS: dict[str, str] = {
    "": "upload.gofile.io",
    "eu-par": "upload-eu-par.gofile.io",
    "na-phx": "upload-na-phx.gofile.io",
    "ap-sgp": "upload-ap-sgp.gofile.io",
    "ap-hkg": "upload-ap-hkg.gofile.io",
    "ap-tyo": "upload-ap-tyo.gofile.io",
    "sa-sao": "upload-sa-sao.gofile.io",
}

# Extension settings exposed to the plugin system
extension_settings = ExtensionInformation(
    extension_type="post_download",
    settings={
        "priority": 200,
        "enabled": True,
        "api_token": "",
        "folder_id": "",
        "compression_level": 0,
        "delete_after_upload": False,
        "upload_region": "",
    },
)


class GofileResponse(msgspec.Struct):
    """Gofile API response structure.

    Attributes:
        status: Response status ("ok" or "error").
        data: Response data dictionary.
    """

    status: str
    data: dict[str, Any] = msgspec.field(default_factory=dict)


class ExtensionSettings(msgspec.Struct, kw_only=True):
    """Settings for Gofile upload extension.

    Attributes:
        enabled: Whether upload is enabled.
        api_token: Gofile API token (optional, uses guest account if empty).
        folder_id: Target folder ID (optional, creates new folder if empty).
        compression_level: ZIP compression level (0-9, 0=store, 9=best).
        delete_after_upload: Whether to delete archive and source files after
            successful upload.
        upload_region: Upload region code (empty for auto).
    """

    enabled: bool = True
    api_token: str = ""
    folder_id: str = ""
    compression_level: int = 0
    delete_after_upload: bool = False
    upload_region: str = ""


def _render_progress_to_string(progress: Progress) -> str:
    """Render rich Progress to a string without ANSI escape codes.

    Args:
        progress: Rich Progress instance.

    Returns:
        Rendered progress bar as plain text string.
    """
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=False, no_color=True, width=80)
    console.print(progress.get_renderable())
    return string_io.getvalue().strip()


class GofileUploader(ExtensionBase):
    """Extension for batch uploading files to Gofile.io.

    This extension collects all download paths and creates a single ZIP
    archive containing all of them, then uploads to Gofile.
    """

    # Class-level storage for collected paths (shared across all calls)
    _collected_paths: ClassVar[list[Path]] = []
    _lock: ClassVar[asyncio.Lock | None] = None
    _finalized: ClassVar[bool] = False

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialize the extension.

        Args:
            settings: Extension configuration dictionary.
        """
        super().__init__(settings)
        self.settings = msgspec.convert(settings, ExtensionSettings)

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create the class-level lock.

        Returns:
            The shared asyncio.Lock instance.
        """
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    async def on_job_complete(self, job: "DownloadJob") -> None:
        """Collect download path for batch processing.

        This method collects paths. The actual upload happens when
        on_all_complete() is called after all downloads complete.

        Args:
            job: The completed download job containing all track information.
        """
        if not self.settings.enabled:
            logger.debug("Gofile upload is disabled")
            return

        if not job.download_path:
            logger.warning("Job has no download path: %s", job.job_id)
            return

        path = Path(job.download_path.rstrip("/\\"))

        if not path.exists():
            logger.warning("Download path does not exist: %s", path)
            return

        async with GofileUploader._get_lock():
            # Avoid duplicates
            if path not in GofileUploader._collected_paths:
                GofileUploader._collected_paths.append(path)
                logger.info("Collected path for Gofile upload: %s", path)

    async def on_all_complete(self) -> None:
        """Upload all collected paths after all downloads complete.

        Creates a ZIP archive of all collected paths and uploads to Gofile.
        """
        async with GofileUploader._get_lock():
            if GofileUploader._finalized or not GofileUploader._collected_paths:
                logger.debug("No paths to upload or already finalized")
                return

            GofileUploader._finalized = True
            paths = GofileUploader._collected_paths.copy()
            GofileUploader._collected_paths.clear()

        # self.settings is already converted to ExtensionSettings in __init__
        ext_settings = self.settings

        if not ext_settings.enabled:
            return

        logger.info("Finalizing Gofile upload with %d paths", len(paths))

        # Check if we have a single file (no need to compress)
        if len(paths) == 1 and paths[0].is_file():
            upload_path = paths[0]
            is_archive = False
            logger.info("Single file detected, uploading directly: %s", upload_path)
        else:
            # Determine output directory (use first path's parent)
            output_dir = paths[0].parent
            # Generate archive name using current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_path = output_dir / f"{timestamp}.zip"
            is_archive = True

            # Create ZIP archive containing all paths
            try:
                logger.info(
                    "Creating ZIP archive: %s with %d items",
                    upload_path,
                    len(paths),
                )
                await asyncio.to_thread(
                    compress_to_zip,
                    paths,
                    upload_path,
                    ext_settings.compression_level,
                )
                logger.info("ZIP archive created successfully: %s", upload_path)
            except OSError as e:
                logger.error("Failed to create ZIP archive: %s", e)
                GofileUploader._finalized = False
                return

        # Upload to Gofile
        result = await self._upload_to_gofile(upload_path, ext_settings)

        if result:
            logger.info("Gofile upload successful: %s", result.get("downloadPage", ""))

            # Clean up after successful upload if requested
            if ext_settings.delete_after_upload and is_archive:
                # Delete the archive file
                if upload_path.exists():
                    await delete_path(upload_path)

                # Delete the source directories/files that were compressed
                for source_path in paths:
                    if source_path.exists():
                        await delete_path(source_path)

        # Reset for next batch
        GofileUploader._finalized = False

    @classmethod
    def reset(cls) -> None:
        """Reset the collector state for a new batch."""
        cls._collected_paths.clear()
        cls._finalized = False

    async def _create_file_sender(
        self,
        file_path: Path,
        file_size: int,
        progress: Progress,
        task_id: TaskID,
    ) -> AsyncIterator[bytes]:
        """Create async generator that yields file chunks with progress.

        Args:
            file_path: Path to the file to read.
            file_size: Total file size in bytes.
            progress: Rich Progress instance.
            task_id: Progress task ID.

        Yields:
            File chunks as bytes.
        """
        uploaded = 0
        last_log_percent = -5  # Log every 5%
        chunk_size = 64 * 1024  # 64KB chunks

        with file_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                uploaded += len(chunk)
                progress.update(task_id, completed=uploaded)

                # Log progress every 5%
                if file_size > 0:
                    current_percent = int(uploaded * 100 / file_size)
                    if current_percent >= last_log_percent + 5:
                        last_log_percent = current_percent
                        progress_str = _render_progress_to_string(progress)
                        self.log(progress_str)

                yield chunk

    def _prepare_upload_form(
        self,
        file_sender: AsyncIterator[bytes],
        filename: str,
        settings: ExtensionSettings,
    ) -> aiohttp.FormData:
        """Prepare multipart form data for upload.

        Args:
            file_sender: Async iterator yielding file chunks.
            filename: Name of the file.
            settings: Extension settings.

        Returns:
            Configured FormData instance.
        """
        form_data = aiohttp.FormData(quote_fields=False)
        form_data.add_field(
            "file",
            file_sender,
            filename=filename,
            content_type="application/octet-stream",
        )

        if settings.folder_id:
            form_data.add_field("folderId", settings.folder_id)

        return form_data

    async def _handle_upload_response(
        self, response: aiohttp.ClientResponse, filename: str
    ) -> dict[str, Any] | None:
        """Handle upload response and parse result.

        Args:
            response: HTTP response object.
            filename: Name of the uploaded file.

        Returns:
            Upload response data if successful, None otherwise.
        """
        if response.status != 200:
            text = await response.text()
            logger.error(
                "Gofile upload failed with status %d: %s",
                response.status,
                text,
            )
            self.log(f"❌ Gofile 上传失败: HTTP {response.status}")
            return None

        content = await response.read()
        result = msgspec.json.decode(content, type=GofileResponse)

        if result.status != "ok":
            logger.error("Gofile upload failed: %s", result.data)
            self.log(f"❌ Gofile 上传失败: {result.data}")
            return None

        download_page = result.data.get("downloadPage", "")
        file_id = result.data.get("fileId", "")
        logger.info(
            "Upload completed - File ID: %s, Download page: %s",
            file_id,
            download_page,
        )
        self.log(f"✅ 上传完成: {download_page}")

        return result.data

    def _handle_upload_error(self, error: Exception, filename: str) -> None:
        """Handle upload errors and log appropriately.

        Args:
            error: Exception that occurred.
            filename: Name of the file being uploaded.
        """
        if isinstance(error, TimeoutError):
            logger.error("Gofile upload timed out for: %s", filename)
            self.log(f"❌ Gofile 上传超时: {filename}")
        elif isinstance(error, aiohttp.ClientError):
            logger.error("Gofile upload HTTP error: %s", error)
            self.log(f"❌ Gofile 上传错误: {error}")

    async def _upload_to_gofile(
        self,
        file_path: Path,
        settings: ExtensionSettings,
    ) -> dict[str, Any] | None:
        """Upload a file to Gofile.io with progress reporting.

        Args:
            file_path: Path to the file to upload.
            settings: Extension settings.

        Returns:
            Upload response data if successful, None otherwise.
        """
        # Get upload endpoint
        region = settings.upload_region
        upload_host = UPLOAD_REGIONS.get(region, UPLOAD_REGIONS[""])
        upload_url = f"https://{upload_host}/uploadfile"

        # Prepare headers
        headers: dict[str, str] = {}
        if settings.api_token:
            headers["Authorization"] = f"Bearer {settings.api_token}"

        # Configure timeout (sock_read=600s, total=None for large file uploads)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=600)

        # Get file size for progress tracking
        file_size = file_path.stat().st_size
        filename = file_path.name

        # Truncate filename for display
        display_name = filename[:27] + "..." if len(filename) > 30 else filename

        try:
            logger.info("Uploading to Gofile: %s", filename)
            self.log(f"开始上传到 Gofile: {filename}")

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Create rich progress for text rendering
                progress = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(binary_units=True),
                    BinaryTransferSpeedColumn(),
                    transient=True,
                )
                task_id: TaskID = progress.add_task(display_name, total=file_size)

                # Create file sender and form data
                file_sender = self._create_file_sender(
                    file_path, file_size, progress, task_id
                )
                form_data = self._prepare_upload_form(file_sender, filename, settings)

                async with session.post(
                    upload_url,
                    data=form_data,
                    headers=headers if headers else None,
                ) as response:
                    return await self._handle_upload_response(response, filename)

        except (TimeoutError, aiohttp.ClientError) as e:
            self._handle_upload_error(e, filename)
            return None
