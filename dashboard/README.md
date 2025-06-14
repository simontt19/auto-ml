# Auto ML Dashboard

A modern, responsive web dashboard for the Auto ML framework with multi-user support.

## Features

- **User Authentication**: Secure login/logout with token-based authentication
- **Project Management**: Create, view, and manage ML projects
- **Experiment Tracking**: Monitor training experiments in real-time
- **Model Management**: View and deploy trained models
- **Monitoring Dashboard**: Real-time model performance and drift detection
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **Frontend**: Next.js 15 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React hooks
- **API Integration**: Custom API client for FastAPI backend
- **Authentication**: Token-based with localStorage

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Auto ML FastAPI backend running (optional for full functionality)

### Installation

1. **Install dependencies**:

   ```bash
   npm install
   ```

2. **Set environment variables** (optional):

   ```bash
   # Create .env.local file
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run the development server**:

   ```bash
   npm run dev
   ```

4. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Usage

### Authentication

The dashboard currently uses demo mode for authentication:

1. Click "Sign in (Demo)" on the login page
2. You'll be logged in as a demo user with sample projects
3. Use the "Sign out" button to log out

### Dashboard Overview

- **Stats Cards**: View total projects, experiments, models, and last updated time
- **Project List**: See all your projects with key metrics
- **Quick Actions**: Start new experiments or create projects

### Project Management

- **Project Details**: Click "View Details" on any project
- **Tabs**: Navigate between Overview, Experiments, Models, and Monitoring
- **Start Experiments**: Use the "Start New Experiment" button
- **Monitor Progress**: Track experiment status and results

### API Integration

The dashboard is designed to integrate with the Auto ML FastAPI backend:

- **Authentication**: Bearer token authentication
- **Project Data**: Real-time project and experiment data
- **Model Management**: Deploy and monitor models
- **Health Monitoring**: System status and performance metrics

## Development

### Project Structure

```
dashboard/
├── src/
│   ├── app/                    # Next.js app router
│   │   ├── page.tsx           # Main dashboard page
│   │   ├── layout.tsx         # Root layout
│   │   └── projects/
│   │       └── [id]/
│   │           └── page.tsx   # Project detail page
│   └── lib/
│       └── api.ts             # API client
├── public/                    # Static assets
└── package.json
```

### Key Components

- **Main Dashboard** (`src/app/page.tsx`): Overview with stats and project list
- **Project Detail** (`src/app/projects/[id]/page.tsx`): Detailed project view
- **API Client** (`src/lib/api.ts`): Communication with FastAPI backend

### Customization

#### Styling

The dashboard uses Tailwind CSS for styling. Key classes:

- `bg-gray-50`: Light gray background
- `bg-white shadow`: Card styling
- `text-blue-600`: Primary blue color
- `rounded-lg`: Rounded corners

#### API Integration

To connect to the FastAPI backend:

1. Set `NEXT_PUBLIC_API_URL` environment variable
2. Update API endpoints in `src/lib/api.ts`
3. Implement real authentication in the API client

## Deployment

### Vercel (Recommended)

1. **Connect to GitHub**:

   ```bash
   git add .
   git commit -m "Add dashboard"
   git push origin main
   ```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set environment variables
   - Deploy

### Other Platforms

The dashboard can be deployed to any platform that supports Next.js:

- **Netlify**: Use `npm run build` and deploy the `out` directory
- **Railway**: Connect GitHub repository and deploy
- **Docker**: Use the provided Dockerfile

## API Endpoints

The dashboard expects the following FastAPI endpoints:

- `GET /health` - Health check
- `GET /models` - List models (with authentication)
- `POST /predict` - Make predictions (with authentication)
- `GET /projects` - List projects (with authentication)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Auto ML framework and follows the same license.

## Support

For issues and questions:

1. Check the [Auto ML documentation](../prompts/)
2. Review the [project state](../prompts/project_state.md)
3. Create an issue in the GitHub repository

## Troubleshooting

1. Check the [Auto ML documentation](../prompts/)
2. Review the [project state](../prompts/project_state.md)
