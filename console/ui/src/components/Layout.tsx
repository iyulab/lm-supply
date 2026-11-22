import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
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
  Wand2,
  Download,
  Loader2,
  RefreshCw,
  Menu,
  X,
  Key,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { useSystemStore } from '../stores/systemStore';

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
  { to: '/image-generate', icon: Wand2, label: 'Image Gen' },
];

function UpdateIndicator() {
  const {
    currentVersion,
    updateCheck,
    updateStatus,
    updateProgress,
    updateError,
    checkForUpdate,
    applyUpdate,
  } = useSystemStore();

  useEffect(() => {
    checkForUpdate();
  }, [checkForUpdate]);

  const statusLabel = () => {
    switch (updateStatus) {
      case 'updating':
        if (updateProgress?.status === 'Downloading')
          return `Downloading ${updateProgress.percent}%`;
        if (updateProgress?.status === 'Extracting') return 'Extracting...';
        if (updateProgress?.status === 'Replacing') return 'Replacing...';
        return 'Updating...';
      case 'restarting':
        return 'Restarting...';
      case 'error':
        return updateError || 'Update failed';
      default:
        return null;
    }
  };

  const isWorking = updateStatus === 'updating' || updateStatus === 'restarting';
  const label = statusLabel();

  return (
    <div className="px-3 py-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          v{currentVersion ?? '...'}
        </span>
        {updateStatus === 'available' && updateCheck && (
          <button
            onClick={applyUpdate}
            className="flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            title={`Update to v${updateCheck.latestVersion}`}
          >
            <Download className="w-3 h-3" />
            v{updateCheck.latestVersion}
          </button>
        )}
      </div>
      {isWorking && (
        <div className="mt-1.5">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {updateStatus === 'restarting' ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Loader2 className="w-3 h-3 animate-spin" />
            )}
            <span>{label}</span>
          </div>
          {updateProgress?.status === 'Downloading' && (
            <div className="mt-1 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${updateProgress.percent}%` }}
              />
            </div>
          )}
        </div>
      )}
      {updateStatus === 'error' && (
        <p className="mt-1 text-xs text-destructive truncate" title={label ?? undefined}>
          {label}
        </p>
      )}
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
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
            onClick={onNavigate}
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
              onClick={onNavigate}
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
            onClick={onNavigate}
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
          <NavLink
            to="/api-keys"
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent hover:text-accent-foreground'
              )
            }
          >
            <Key className="w-4 h-4" />
            API Keys
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
          {/* Version & Update */}
          <div className="border-t border-border mt-1 pt-1">
            <UpdateIndicator />
          </div>
        </div>
      </nav>
    </>
  );
}

export function Layout() {
  const location = useLocation();
  // The sidebar is open for the route it was opened on, so a route change closes it (mobile) without an effect
  const [openedOn, setOpenedOn] = useState<string | null>(null);
  const sidebarOpen = openedOn === location.pathname;
  const setSidebarOpen = (open: boolean) => setOpenedOn(open ? location.pathname : null);

  return (
    <div className="flex h-screen bg-background">
      {/* Mobile top bar */}
      <div className="fixed top-0 left-0 right-0 z-30 flex items-center gap-3 p-3 border-b border-border bg-card md:hidden">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-1.5 rounded-md hover:bg-accent"
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
        <span className="font-semibold text-sm">LMSupply Console</span>
      </div>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — desktop: always visible, mobile: slide-in overlay */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 w-64 border-r border-border bg-card transition-transform duration-200 ease-in-out md:static md:translate-x-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <SidebarContent onNavigate={() => setSidebarOpen(false)} />
      </aside>

      {/* Main Content — add top padding on mobile for the fixed top bar */}
      <main className="flex-1 overflow-auto pt-12 md:pt-0">
        <Outlet />
      </main>
    </div>
  );
}
