import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Cat, History, Zap } from 'lucide-react';
import { HomePage } from './pages/Home';
import { HistoryPage } from './pages/History';
import './index.css';

function NavBar() {
  const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 16px',
    borderRadius: 10,
    fontSize: 14,
    fontWeight: 600,
    textDecoration: 'none',
    transition: 'all 0.2s ease',
    color: isActive ? 'var(--color-accent)' : 'var(--color-text-muted)',
    background: isActive ? 'rgba(139,92,246,0.12)' : 'transparent',
    border: isActive ? '1px solid rgba(139,92,246,0.25)' : '1px solid transparent',
  });

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      background: 'rgba(10,11,20,0.85)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(139,92,246,0.1)',
    }}>
      <div style={{
        maxWidth: 1100,
        margin: '0 auto',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 60,
        gap: 8,
      }}>
        {/* Logo */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          marginRight: 24,
        }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(139,92,246,0.4)',
          }}>
            <Cat size={18} color="white" />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, lineHeight: 1, letterSpacing: '-0.02em' }}>
              <span style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}>
                Pet
              </span>
              {' '}Classifier
            </div>
            <div style={{ fontSize: 10, color: 'var(--color-text-faint)', lineHeight: 1, marginTop: 2 }}>
              Powered by CLIP
            </div>
          </div>
        </div>

        {/* Nav links */}
        <NavLink to="/" style={navLinkStyle} end id="nav-home">
          <Zap size={15} />
          Phân loại
        </NavLink>

        <NavLink to="/history" style={navLinkStyle} id="nav-history">
          <History size={15} />
          Lịch sử
        </NavLink>
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
