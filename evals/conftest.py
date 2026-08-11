from pathlib import Path

from dotenv import load_dotenv

# Loaded once for the whole evals/ pytest session, same as interface/cli.py does
# for the real CLI — otherwise GROQ_API_KEY from .env is invisible to the
# end-to-end agent suite (evals/test_agent_evals.py) when run via pytest.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
