import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageSquare,
  Database,
  Mic,
  Volume2,
  Search,
  Settings,
  Image,
  ScanText,
  ScanSearch,
  Grid3X3,
  Languages,
  FileText,
  ExternalLink,
} from 'lucide-react';
import { cn } from '../lib/utils';

// Top navigation items
const topNavItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
];

// Feature navigation items
const featureNavItems = [
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/embed', icon: Database, label: 'Embed' },
  { to: '/rerank', icon: Search, label: 'Rerank' },
  { to: '/transcribe', icon: Mic, label: 'Transcribe' },
  { to: '/synthesize', icon: Volume2, label: 'Synthesize' },
  { to: '/caption', icon: Image, label: 'Caption' },
  { to: '/ocr', icon: ScanText, label: 'OCR' },
  { to: '/detect', icon: ScanSearch, label: 'Detect' },
  { to: '/segment', icon: Grid3X3, label: 'Segment' },
  { to: '/translate', icon: Languages, label: 'Translate' },
];

export function Layout() {
  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border bg-card">
        <div className="p-4 border-b border-border">
          <h1 className="text-xl font-bold">LMSupply Console</h1>
          <p className="text-sm text-muted-foreground">Local AI Testing</p>
        </div>
        <nav className="p-2 flex flex-col h-[calc(100%-5rem)]">
          {/* Dashboard */}
          {topNavItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-accent hover:text-accent-foreground'
                )
              }
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </NavLink>
          ))}

          {/* Divider */}
          <div className="border-t border-border my-2" />

          {/* Feature items */}
          <div className="flex-1 overflow-y-auto">
            {featureNavItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-accent hover:text-accent-foreground'
                  )
                }
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </NavLink>
            ))}
          </div>

          {/* Bottom section: Models & API Docs */}
          <div className="border-t border-border pt-2 mt-2 space-y-1">
            <NavLink
              to="/models"
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-accent hover:text-accent-foreground'
                )
              }
            >
              <Settings className="w-4 h-4" />
              Models
            </NavLink>
            <a
              href="/swagger"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              <FileText className="w-4 h-4" />
              API Docs
              <ExternalLink className="w-3 h-3 ml-auto opacity-50" />
            </a>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
