"""Email delivery service - entry point and wiring."""

from .config import EmailConfig
from .provider import EmailProvider, ResendProvider, SMTPProvider, MockProvider, EmailMessage, SendResult
from .renderer import EmailRenderer
from .tracker import DeliveryTracker
from .retry import RetryHandler
from .sender import EmailSender

# init_email_service (which registered routes/alerts.py and a digest scheduler)
# was removed 2026-11-25: the routes had no authentication and the scheduler's
# jobs crashed on a schema the live DB never had. The provider/renderer classes
# above remain importable.
