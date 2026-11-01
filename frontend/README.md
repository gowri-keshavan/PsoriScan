# PsoriScan frontend

React app (Create React App) for the PsoriScan API.

```bash
cp .env.example .env     # set REACT_APP_API_URL to your API
npm install
npm start                # http://localhost:3000
npm test -- --watchAll=false
npm run build            # production build in build/
```

| Variable | Default | Purpose |
|---|---|---|
| `REACT_APP_API_URL` | `http://localhost:5000` | Base URL of the API |
| `REACT_APP_MAX_UPLOAD_MB` | `8` | Client-side upload limit (keep equal to the API's `MAX_UPLOAD_MB`) |

Environment variables are baked in at build time, so set them in your hosting provider before building.
