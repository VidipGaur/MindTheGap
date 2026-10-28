# Copyright 2026 Vidip Gaur, Vidu Gaur
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
try:
    me = api.whoami()
    print("account :", me.get("name"), "| type:", me.get("type"))
    auth = me.get("auth", {}).get("accessToken", {})
    print("token   :", auth.get("displayName"), "| role:", auth.get("role"))
except Exception as e:
    print("whoami FAILED:", type(e).__name__, e)

print()
for repo in ["MahmoodLab/UNI2-h", "bioptimus/H-optimus-0", "histai/hibou-L",
             "MahmoodLab/UNI", "owkin/phikon-v2"]:
    try:
        api.hf_hub_download(repo_id=repo, filename="config.json")
        print("%-24s OK  (config.json fetched)" % repo)
    except Exception as e:
        msg = str(e).splitlines()[0][:110]
        print("%-24s %s: %s" % (repo, type(e).__name__, msg))
