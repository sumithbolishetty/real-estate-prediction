# Frontend

Next.js with TypeScript and Tailwind CSS. The form obtains its locations, property types, parking options, and amenity choices from the backend. It uses Indian number grouping for the displayed estimate.

From project root, run `cd frontend`, `cp .env.local.example .env.local`, `npm install`, then `npm run dev`. The app opens at `http://localhost:3100`. Set `NEXT_PUBLIC_API_URL` in `.env.local` to the backend URL when changing local setup; the documented default is `http://127.0.0.1:8100`. The dev script explicitly sets the frontend port.
