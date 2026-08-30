"""
🎨 Shiny UI Styling Module - Professional Navy Blue Medical Analytics Theme

⚠️ SOURCE OF TRUTH FOR CSS:
Modify this file to update 'static/styles.css'.
After editing, run: python utils/update_css.py

Provides comprehensive CSS utilities, styled components, and helper functions for consistent
UI styling across all Shiny modules using a professional Navy Blue theme with enhanced
visual hierarchy, spacing, and modern design patterns.

Features:
- Professional Navy Blue color palette with medical-grade aesthetics
- Enhanced card designs with better shadows and hover effects
- Modern button styles with improved accessibility
- Refined form inputs with better focus states
- Improved spacing and typography hierarchy
- Responsive design optimizations
- Better status indicators and badges
- Enhanced table styling

Usage:
    from tabs._styling import get_shiny_css, style_card_header, style_status_badge, style_alert

    ui.HTML(get_shiny_css())
"""

from tabs._common import get_color_palette


def get_shiny_css():
    """
    Provide the full CSS stylesheet implementing the Navy Blue theme for Shiny apps.

    Includes root variables (colors, spacing, radii, shadows, transitions, typography), Bootstrap-compatible component styles (.card, .card-header, .btn, .form-control, .nav-tabs, etc.), utility classes, custom components, and responsive adjustments.

    Returns:
        css (str): An HTML string containing a <style>...</style> block with the complete CSS for the theme.
    """
    COLORS = get_color_palette()

    css = f"""
    <style>
        /* ===========================
           CSS VARIABLES & ROOT STYLES
           =========================== */
        
        :root {{
            --color-primary: {COLORS["primary"]};
            --color-primary-dark: {COLORS["primary_dark"]};
            --color-primary-light: {COLORS["primary_light"]};
            --color-smoke-white: {COLORS["smoke_white"]};
            --color-success: {COLORS["success"]};
            --color-danger: {COLORS["danger"]};
            --color-warning: {COLORS["warning"]};
            --color-info: {COLORS["info"]};
            --color-neutral: {COLORS["neutral"]};
            --color-text: {COLORS["text"]};
            --color-text-secondary: {COLORS["text_secondary"]};
            --color-border: {COLORS["border"]};
            --color-background: {COLORS["background"]};
            --color-surface: {COLORS["surface"]};
            
            /* ADDED: Semantic States (Audit Section 9) */
            --color-modified: #8B5CF6;    /* Purple - user has modified */
            --color-processing: #3B82F6;  /* Blue - computation running */
            --color-valid: #10B981;       /* Green - validation passed */
            --color-attention: #F59E0B;   /* Amber - needs attention */
            
            /* ADDED: Feedback States (Audit Section 9) */
            --color-step-complete: #10B981;
            --color-step-current: #1E3A5F;
            --color-step-pending: #D1D5DB;

            /* Spacing System */
            --spacing-2xs: 2px;
            --spacing-1-5xs: 6px;
            --spacing-xs: 4px;
            --spacing-sm: 8px;
            --spacing-1-5sm: 12px;
            --spacing-md: 16px;
            --spacing-lg: 20px;
            --spacing-xl: 32px;
            --spacing-2xl: 48px;
            
            /* Component-Specific Spacing */
            --spacing-card-vertical: 24px;
            --spacing-section-vertical: 32px;
            --spacing-input-gap: 8px;
            --spacing-form-section: 20px;
            
            /* Border Radius - Crisper edges */
            --radius-sm: 4px;
            --radius-md: 6px;
            --radius-lg: 8px;
            --radius-xl: 12px;
            
            /* Shadows - Flattened for minimal aesthetic */
            --shadow-sm: none;
            --shadow-md: 0 1px 3px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 4px 12px rgba(0, 0, 0, 0.06);
            --shadow-xl: 0 8px 24px rgba(0, 0, 0, 0.08);
            
            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            
            /* Typography */
            --font-family-base: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            --font-family-mono: 'Courier New', monospace;
            
            /* Typography Scale */
            --text-display-size: 30px;
            --text-display-lh: 1.2;
            --text-heading-size: 22px;
            --text-heading-lh: 1.3;
            --text-subheading-size: 16px;
            --text-subheading-lh: 1.4;
            --text-body-size: 14px;
            --text-body-lh: 1.6;
            --text-caption-size: 12px;
            --text-caption-lh: 1.5;
            --text-code-size: 13px;
            --text-code-lh: 1.4;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        .report-footer {{
            text-align: center;
            padding: var(--spacing-xl) 0;
            margin-top: var(--spacing-2xl);
            border-top: 1px solid {COLORS["border"]};
            color: {COLORS["text_secondary"]};
            font-size: 13px;
            background-color: {COLORS["surface"]};
        }}
        
        .report-footer a {{
            color: {COLORS["primary"]};
            text-decoration: none;
            font-weight: 600;
        }}
        
        .report-footer a:hover {{
            text-decoration: underline;
        }}
        
        /* ===========================
           LAYOUT & CONTAINER
           =========================== */
        
        /* Main app container - centered and max-width on larger screens */
        .app-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 32px 48px; /* Increased Padding */
        }}
        
        /* Add top margin to app-container when it follows navbar */
        .navbar ~ .app-container {{
            margin-top: 16px;
        }}
        
        /* Responsive padding on mobile */
        @media (max-width: 768px) {{
            .app-container {{
                padding: 16px 20px 32px;
            }}
        }}
        
        /* ===========================
           GLOBAL STYLES
           =========================== */
        
        html {{
            font-size: 14px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        body {{
            font-family: var(--font-family-base);
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        
        /* ===========================
           TYPOGRAPHY
           =========================== */
        
        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin: var(--spacing-lg) 0;
            color: {COLORS["primary_dark"]};
            letter-spacing: -0.5px;
        }}
        
        h2 {{
            font-size: 24px;
            font-weight: 600;
            margin: var(--spacing-lg) 0 var(--spacing-md) 0;
            color: {COLORS["primary_dark"]};
            letter-spacing: -0.3px;
        }}
        
        h3 {{
            font-size: 20px;
            font-weight: 600;
            margin: var(--spacing-md) 0;
            color: {COLORS["primary"]};
        }}
        
        h4 {{
            font-size: 17px;
            font-weight: 600;
            margin: var(--spacing-md) 0;
            color: {COLORS["primary"]};
        }}
        
        h5 {{
            font-size: 15px;
            font-weight: 600;
            margin: var(--spacing-sm) 0;
            color: {COLORS["text"]};
        }}
        
        h6 {{
            font-size: 13px;
            font-weight: 600;
            margin: var(--spacing-sm) 0;
            color: {COLORS["text_secondary"]};
        }}
        
        p {{
            margin: 0 0 var(--spacing-md) 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        a {{
            color: {COLORS["primary"]};
            text-decoration: none;
            transition: color var(--transition-fast);
            font-weight: 500;
        }}
        
        a:hover {{
            color: {COLORS["primary_dark"]};
            text-decoration: underline;
        }}
        
        /* ===========================
           SHINY CARDS & CONTAINERS
           (Updated to standard Bootstrap .card classes)
           =========================== */
          .card {{
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm); /* None */
            transition: all var(--transition-normal);
            background-color: {COLORS["surface"]};
            overflow: hidden;
            margin-bottom: 24px;
        }}
        
        .card:hover {{
            box-shadow: none;
            border-color: {COLORS["neutral"]};
        }}
        
        /* Feature Cards (Home Page) */
        .feature-card {{
            background: white !important;
            border: 1px solid {COLORS["border"]} !important;
            border-radius: var(--radius-lg) !important;
            padding: 24px !important;
            height: 100%;
            transition: border-color var(--transition-normal) !important;
            cursor: pointer;
            box-shadow: none !important;
            transform: none !important;
        }}
        
        .feature-card:hover {{
            border-color: {COLORS["primary"]} !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        
        .card-header {{
            background: {COLORS["background"]}; /* Transparent fallback appeal */
            border: none;
            border-bottom: 1px solid {COLORS["border"]};
            font-weight: 500;
            color: {COLORS["primary"]};
            padding: 14px 20px;
            font-size: 14px;
            text-transform: none;
            display: flex;
            align-items: center;
        }}
        
        .card-body {{
            padding: 20px 20px;
            line-height: 1.6;
        }}
        
        .card-footer {{
            background-color: {COLORS["background"]};
            border-top: 1px solid {COLORS["border"]};
            padding: 14px 20px;
        }}
        
        /* ===========================
           BUTTONS - Flat & Modern
           =========================== */
        
        .btn {{
            border: 1px solid transparent;
            border-radius: var(--radius-md);
            font-weight: 500;
            font-size: 14px;
            padding: 8px 16px;
            transition: all var(--transition-fast);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: var(--spacing-sm);
            white-space: nowrap;
            user-select: none;
            box-shadow: none; /* Removed heavy shadows */
        }}
        
        .btn:focus {{
            outline: 2px solid {COLORS["primary"]};
            outline-offset: 2px;
        }}
        
        .btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            pointer-events: none;
        }}

        /* ADDED: Button Size Variants (Audit Section 9) */
        .btn-sm {{
            padding: 4px 12px;
            font-size: 13px;
        }}
        
        .btn-lg {{
            padding: 12px 24px;
            font-size: 16px;
        }}
        
        /* ADDED: Button State Variants (Audit Section 9) */
        .btn-loading {{
            position: relative;
            color: transparent !important;
            pointer-events: none;
        }}
        
        .btn-loading::after {{
            content: "";
            position: absolute;
            width: 16px;
            height: 16px;
            top: 50%;
            left: 50%;
            margin-top: -8px;
            margin-left: -8px;
            border: 2px solid rgba(255,255,255,0.5);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }}
        
        .btn-outline {{
            background-color: transparent;
            border: 1px solid {COLORS["border"]};
            color: {COLORS["text"]};
        }}
        
        .btn-outline:hover {{
            border-color: {COLORS["primary"]};
            color: {COLORS["primary"]};
            background-color: {COLORS["smoke_white"]};
        }}
        
        /* Primary Buttons - Flat */
        .btn-primary {{
            background-color: {COLORS["primary"]};
            color: white;
            border-color: {COLORS["primary"]};
        }}
        
        .btn-primary:hover:not(:disabled) {{
            background-color: {COLORS["primary_dark"]};
            border-color: {COLORS["primary_dark"]};
            transform: none;
        }}
        
        .btn-primary:active:not(:disabled) {{
            transform: none;
        }}
        
        /* Secondary Buttons - Outline style mostly */
        .btn-secondary {{
            background-color: transparent;
            color: {COLORS["text"]};
            border-color: {COLORS["border"]};
        }}
        
        .btn-secondary:hover:not(:disabled) {{
            background-color: {COLORS["smoke_white"]};
            border-color: {COLORS["neutral"]};
            color: {COLORS["primary"]};
        }}
        
        /* Success Buttons */
        .btn-success {{
            background-color: {COLORS["success"]};
            color: white;
            border-color: {COLORS["success"]};
        }}
        
        .btn-success:hover:not(:disabled) {{
            filter: brightness(0.9);
            transform: none;
        }}
        
        /* Danger Buttons */
        .btn-danger {{
            background-color: {COLORS["danger"]};
            color: white;
            border-color: {COLORS["danger"]};
        }}
        
        .btn-danger:hover:not(:disabled) {{
            filter: brightness(0.9);
            transform: none;
        }}
        
        /* Warning Buttons */
        .btn-warning {{
            background-color: {COLORS["warning"]};
            color: #000;
            border-color: {COLORS["warning"]};
        }}
        
        /* Outline Buttons */
        .btn-outline-primary {{
            border: 1px solid {COLORS["primary"]};
            color: {COLORS["primary"]};
            background-color: transparent;
        }}
        
        .btn-outline-primary:hover:not(:disabled) {{
            background-color: {COLORS["primary_light"]};
            color: {COLORS["primary_dark"]};
        }}
        
        /* ===========================
           FORM INPUTS - Stripe/Apple-like
           =========================== */
        
        .form-control {{
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-md);
            padding: 10px 14px;
            font-size: 14px;
            font-family: var(--font-family-base);
            background-color: {COLORS["primary_light"]}; /* Slate 50 tint */
            color: {COLORS["text"]};
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
            line-height: 1.5;
        }}
        
        .form-control:focus {{
            border-color: {COLORS["primary"]};
            background-color: {COLORS["surface"]};
            box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.15); /* Soft slate ring */
            outline: none;
        }}
        
        .form-control::placeholder {{
            color: {COLORS["text_secondary"]};
            opacity: 0.7;
        }}
        
        .form-control:disabled,
        .form-select:disabled {{
            background-color: {COLORS["background"]};
            border-color: {COLORS["border"]};
            color: {COLORS["text_secondary"]};
            cursor: not-allowed;
            opacity: 0.8;
        }}
        
        /* Select/Dropdown */
        .form-select {{
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-md);
            padding: 10px 36px 10px 14px;
            font-size: 14px;
            background-color: {COLORS["primary_light"]}; /* Slate 50 tint */
            color: {COLORS["text"]};
            transition: all var(--transition-fast);
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{COLORS["text_secondary"].replace("#", "%23")}' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
            background-size: 16px;
        }}
        
        .form-select:focus {{
            border-color: {COLORS["primary"]};
            background-color: {COLORS["surface"]};
            box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.15); /* Soft slate ring */
            outline: none;
        }}
        
        /* Form Label */
        .form-label {{
            font-weight: 400; /* Muted weight */
            font-size: 13px;
            color: {COLORS["text_secondary"]};
            margin-bottom: 6px;
            display: block;
        }}
        
        .form-label.required::after {{
            content: "";
            display: none;
        }}
        
        /* Form Group - Better Separation */
        .form-group {{
            margin-bottom: 24px;
        }}
        
        /* Textarea */
        textarea.form-control {{
            resize: vertical;
            min-height: 100px;
            font-family: var(--font-family-base);
        }}
          /* ===========================
           NAVIGATION & NAVBAR
           =========================== */
        
        .navbar {{
            background-color: {COLORS["surface"]} !important;
            box-shadow: none; /* No shadow */
            border-bottom: 1px solid {COLORS["border"]};
            padding: 12px 24px;
        }}
        
        .navbar-brand {{
            color: {COLORS["primary"]} !important;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: -0.5px;
        }}
        
        /* Navbar Text & Links */
        .navbar .nav-link {{
            color: {COLORS["text_secondary"]} !important;
            font-weight: 500;
            font-size: 14px;
            padding: 8px 12px !important;
            margin: 0 4px;
            border-radius: 0;
            transition: all var(--transition-fast);
            border-bottom: 2px solid transparent;
        }}
        
        .navbar .nav-link:hover {{
            color: {COLORS["primary"]} !important;
            background-color: transparent;
        }}
        
        .navbar .nav-link.active {{
            color: {COLORS["primary"]} !important;
            background-color: transparent !important;
            font-weight: 500;
            border-bottom: 2px solid {COLORS["primary"]} !important;
        }}
        
        /* Hide Home tab from navbar links (accessible via title) */
        .navbar-nav .nav-link[data-value="home"] {{
            display: none !important;
        }}
        
        /* Dropdown Menu (nav_menu) Styles */
        .navbar .dropdown-toggle {{
            color: {COLORS["text_secondary"]} !important;
            font-weight: 500;
            font-size: 14px;
            padding: 8px 12px !important;
            margin: 0 4px;
            border-radius: 0;
            transition: all var(--transition-fast);
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
        }}
        
        .navbar .dropdown-toggle:hover,
        .navbar .dropdown-toggle:focus {{
            color: {COLORS["primary"]} !important;
            background-color: transparent;
        }}
        
        .navbar .dropdown-toggle::after {{
            margin-left: 6px;
            vertical-align: middle;
        }}
        
        .navbar .dropdown-menu {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            padding: 6px 0;
            min-width: 200px;
            margin-top: 4px;
        }}
        
        .navbar .dropdown-item {{
            color: {COLORS["text_secondary"]};
            font-size: 14px;
            font-weight: 500;
            padding: 8px 20px;
            transition: all var(--transition-fast);
        }}
        
        .navbar .dropdown-item:hover {{
            background-color: {COLORS["smoke_white"]};
            color: {COLORS["primary"]};
        }}
        
        .navbar .dropdown-item.active,
        .navbar .dropdown-item:active {{
            background-color: transparent;
            color: {COLORS["primary"]};
            font-weight: 600;
        }}
        
        /* ===========================
           TABS
           =========================== */
        
        .nav-tabs {{
            border-bottom: 1px solid {COLORS["border"]};
            margin-bottom: 24px;
        }}
        
        .nav-tabs .nav-link {{
            color: {COLORS["text_secondary"]};
            border: none;
            border-bottom: 2px solid transparent;
            font-weight: 500;
            padding: 10px 16px;
            margin-bottom: -1px;
            transition: all var(--transition-fast);
            font-size: 14px;
        }}
        
        .nav-tabs .nav-link:hover {{
            border-bottom-color: {COLORS["neutral"]};
            color: {COLORS["primary"]};
            background-color: transparent;
        }}
        
        .nav-tabs .nav-link.active {{
            color: {COLORS["primary"]};
            background-color: transparent;
            border-bottom-color: {COLORS["primary"]};
            font-weight: 500;
        }}
        
        /* Subtabs - Pills style often mostly used inside spacing */
        .nav-item .nav-link[role="tab"]:not(.active):hover {{
            color: {COLORS["primary"]} !important;
            background-color: transparent;
        }}
        
        /* ===========================
           ALERTS & NOTIFICATIONS
           =========================== */
        
        .alert {{
            border-radius: var(--radius-lg);
            border: 1px solid;
            padding: 12px 16px;
            margin-bottom: 24px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            box-shadow: none;
        }}
        
        .alert-success {{
            background-color: #ECFDF5;
            border-color: #A7F3D0;
            color: #065F46;
        }}
        
        .alert-danger {{
            background-color: #FEF2F2;
            border-color: #FECACA;
            color: #991B1B;
        }}
        
        .alert-warning {{
            background-color: #FFFBEB;
            border-color: #FDE68A;
            color: #92400E;
        }}
        
        .alert-info {{
            background-color: {COLORS["primary_light"]};
            border-color: {COLORS["neutral"]};
            color: {COLORS["info"]};
        }}
        
        /* ===========================
           TABLES
           =========================== */
        
        .table {{
            font-size: 13px;
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            margin-bottom: 24px;
            border: none; /* No heavy border wrapper */
            border-radius: 0;
            overflow: visible;
        }}
        
        .table thead th {{
            background-color: transparent; /* Clean transparent */
            color: {COLORS["text_secondary"]};
            border-bottom: 2px solid {COLORS["border"]};
            font-weight: 500;
            padding: 10px 16px;
            text-align: left;
            text-transform: none; /* Minimal case */
            font-size: 13px;
            letter-spacing: normal;
        }}
        
        .table tbody td {{
            border-bottom: 1px solid {COLORS["border"]};
            padding: 10px 16px;
            vertical-align: middle;
            color: {COLORS["text"]};
        }}
        
        .table tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        .table tbody tr:hover {{
            background-color: {COLORS["primary_light"]};
        }}
        
        .sig-p {{
            color: #fff;
            background-color: {COLORS["danger"]};
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}

        .shiny-table th, table.dataframe th {{
            background-color: transparent !important;
            color: {COLORS["text_secondary"]} !important;
            font-weight: 500;
            border-bottom: 2px solid {COLORS["border"]} !important;
            text-transform: none !important;
        }}
        
        /* Table Responsiveness */
        .table-responsive {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: var(--radius-lg);
            border: 1px solid {COLORS["border"]};
        }}

        .table-responsive .table {{
            border: none;
            border-radius: 0;
            margin-bottom: 0;
        }}
        
        /* Error Cell Highlighting */
        .cell-error {{
            background-color: #FEF2F2 !important;
            color: {COLORS["danger"]} !important;
            font-weight: 600 !important;
        }}
        
        /* ===========================
           BADGES & STATUS INDICATORS
           =========================== */
        
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        /* Status Badge (Custom) */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px; /* Pill shape */
            font-weight: 600;
            font-size: 12px;
            border: 1px solid;
        }}
        
        .status-badge.matched {{
            background-color: #ECFDF5;
            color: #065F46;
            border-color: #A7F3D0;
        }}
        
        .status-badge.unmatched {{
            background-color: #FEF2F2;
            color: #991B1B;
            border-color: #FECACA;
        }}
        
        .status-badge.warning {{
            background-color: #FFFBEB;
            color: #92400E;
            border-color: #FDE68A;
        }}

        /* ===========================
           FEEDBACK & LOADING STATES
           =========================== */
        
        /* Enhanced Loading State */
        .loading-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 48px;
            color: {COLORS["text_secondary"]};
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: var(--radius-lg);
            width: 100%;
            height: 100%;
            min-height: 200px;
        }}

        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid {COLORS["smoke_white"]};
            border-top: 3px solid {COLORS["primary"]};
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Placeholder State */
        .placeholder-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 48px;
            color: {COLORS["text_secondary"]};
            background-color: {COLORS["smoke_white"]};
            border: 2px dashed {COLORS["border"]};
            border-radius: var(--radius-lg);
            text-align: center;
            min-height: 200px;
        }}
        
        .placeholder-icon {{
            font-size: 32px;
            margin-bottom: 12px;
            opacity: 0.5;
        }}

        /* Validation & Error Messages */
        .validation-error {{
            color: {COLORS["danger"]};
            font-size: 13px;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .error-alert-card {{
            border-left: 4px solid {COLORS["danger"]};
            background-color: #FEF2F2;
        }}
        
        /* ===========================
           ACCESSIBILITY
           =========================== */

        /* Skip Navigation Links */
        .skip-links {{
            position: absolute;
            top: -9999px;
            left: -9999px;
            z-index: 9999;
        }}

        .skip-link:focus {{
            position: fixed;
            top: 0;
            left: 0;
            width: auto;
            height: auto;
            background: {COLORS["primary"]};
            color: white;
            padding: 12px 20px;
            text-decoration: none;
            font-weight: 600;
            border-radius: 0 0 var(--radius-md) 0;
            outline: 3px solid {COLORS["warning"]};
            box-shadow: var(--shadow-xl);
        }}

        /* Enhanced Focus Indicators */
        *:focus-visible {{
            outline: 3px solid {COLORS["primary"]};
            outline-offset: 2px;
            border-radius: 2px;
        }}

        /* Screen Reader Only */
        .sr-only {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }}

        /* ===========================
           UTILITY CLASSES
           =========================== */
        
        .text-primary {{
            color: {COLORS["primary"]} !important;
        }}
        
        .border-primary {{
             border: 1px solid {COLORS["primary"]} !important;
             border-radius: var(--radius-md);
        }}
        
        .border-top {{
            border-top: 2px solid {COLORS["border"]} !important;
            padding-top: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }}
        
        .divider {{
            height: 1px;
            background-color: {COLORS["border"]};
            margin: var(--spacing-lg) 0;
        }}
        
        /* ===========================
           VIF SPECIFIC STYLES
           =========================== */
        
        .vif-container {{
            margin-top: var(--spacing-lg);
            padding: var(--spacing-md);
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            max-width: 500px;
        }}
        
        .vif-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--color-primary-dark);
            margin-bottom: var(--spacing-sm);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .vif-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .vif-table th {{
            text-align: left;
            padding: var(--spacing-sm);
            background-color: var(--color-smoke-white);
            color: var(--color-text-secondary);
            font-weight: 600;
            border-bottom: 2px solid var(--color-border);
        }}
        
        .vif-table td {{
            padding: var(--spacing-sm);
            border-bottom: 1px solid var(--color-border);
        }}
        
        .vif-value {{
            font-family: var(--font-family-mono);
            font-weight: 600;
        }}
        
        .vif-warning {{
            color: var(--color-danger);
            font-weight: 700;
        }}
        
        .vif-caution {{
            color: var(--color-warning);
        }}
        
        .vif-footer {{
            margin-top: var(--spacing-sm);
            font-size: 11px;
            color: var(--color-text-secondary);
            font-style: italic;
        }}
        .mt-1 {{ margin-top: var(--spacing-xs); }}
        .mt-2 {{ margin-top: var(--spacing-sm); }}
        .mt-3 {{ margin-top: var(--spacing-md); }}
        .mt-4 {{ margin-top: var(--spacing-lg); }}
        .mt-5 {{ margin-top: var(--spacing-xl); }}
        
        .mb-1 {{ margin-bottom: var(--spacing-xs); }}
        .mb-2 {{ margin-bottom: var(--spacing-sm); }}
        .mb-3 {{ margin-bottom: var(--spacing-md); }}
        .mb-4 {{ margin-bottom: var(--spacing-lg); }}
        .mb-5 {{ margin-bottom: var(--spacing-xl); }}
        
        .p-2 {{ padding: var(--spacing-sm); }}
        .p-3 {{ padding: var(--spacing-md); }}
        .p-4 {{ padding: var(--spacing-lg); }}
        .p-5 {{ padding: var(--spacing-xl); }}
        
        /* ===========================
           CUSTOM COMPONENTS
           =========================== */
        
        .stat-box {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            text-align: center;
            box-shadow: var(--shadow-sm);
            transition: all var(--transition-normal);
        }}
        
        .stat-box:hover {{
            box-shadow: var(--shadow-md);
            border-color: {COLORS["primary_light"]};
        }}
        
        .stat-box-label {{
            font-size: 12px;
            font-weight: 600;
            color: {COLORS["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: var(--spacing-sm);
        }}
        
        .stat-box-value {{
            font-size: 28px;
            font-weight: 700;
            color: {COLORS["primary_dark"]};
            line-height: 1;
        }}
        
        .stat-box-subtext {{
            font-size: 12px;
            color: {COLORS["text_secondary"]};
            margin-top: var(--spacing-sm);
        }}
        
        /* Info Box / Panel */
        .info-panel {{
            background-color: {COLORS["primary_light"]};
            border-left: 4px solid {COLORS["primary"]};
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }}
        
        .info-panel.success {{
            background-color: rgba(34, 167, 101, 0.1);
            border-left-color: {COLORS["success"]};
        }}
        
        .info-panel.danger {{
            background-color: rgba(231, 72, 86, 0.1);
            border-left-color: {COLORS["danger"]};
        }}
        
        .info-panel.warning {{
            background-color: rgba(255, 185, 0, 0.1);
            border-left-color: {COLORS["warning"]};
        }}
        
        /* Data Grid Container */
        .data-grid {{
            overflow-x: auto;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
        }}
        
        /* ===========================
           MISSING DATA SECTION
           =========================== */
        
        .missing-data-section {{
            background-color: {COLORS["primary_light"]};
            border-left: 4px solid {COLORS["primary"]};
            padding: var(--spacing-lg);
            margin: var(--spacing-xl) 0;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
        }}
        
        .missing-data-section h4 {{
            color: {COLORS["primary_dark"]};
            margin-top: 0;
            margin-bottom: var(--spacing-md);
        }}
        
        .missing-data-section h5 {{
            color: {COLORS["text"]};
            margin-top: var(--spacing-lg);
            margin-bottom: var(--spacing-sm);
        }}
        
        .missing-table {{
            width: 100%;
            border-collapse: collapse;
            margin: var(--spacing-md) 0;
            font-size: 13px;
        }}
        
        .missing-table th,
        .missing-table td {{
            padding: var(--spacing-sm) var(--spacing-md);
            text-align: left;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        
        .missing-table th {{
            background-color: {COLORS["primary_light"]};
            font-weight: 600;
            color: {COLORS["primary_dark"]};
        }}
        
        .missing-table tr.high-missing {{
            background-color: #fff3cd;
        }}
        
        .missing-table tr.high-missing td {{
            color: #856404;
            font-weight: 500;
        }}
        
        .warning-box {{
            background-color: #fff3cd;
            border: 1px solid {COLORS["warning"]};
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
            margin-top: var(--spacing-md);
        }}
        
        .warning-box strong {{
            color: #856404;
        }}
        
        .warning-box ul {{
            margin: var(--spacing-sm) 0 0 var(--spacing-lg);
            padding: 0;
        }}
        
        .warning-box li {{
            color: #856404;
            margin-bottom: var(--spacing-xs);
        }}
        
        /* Missing config column styling */
        .variable-config-column {{
            padding-right: var(--spacing-md);
            border-right: 1px solid {COLORS["border"]};
        }}
        
        .missing-config-column {{
            background-color: {COLORS["background"]};
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
        }}
        
        .missing-config-column h5,
        .missing-config-column h6 {{
            color: {COLORS["primary"]};
            margin-bottom: var(--spacing-md);
        }}
        
        /* ===========================
           RESPONSIVENESS
           =========================== */
        
        @media (max-width: 768px) {{
            :root {{
                --spacing-lg: 12px;
                --spacing-xl: 16px;
            }}
            
            .card-body {{
                padding: 12px 14px;
            }}
            
            .btn {{
                font-size: 12px;
                padding: 8px 12px;
            }}
            
            .form-control,
            .form-select {{
                font-size: 16px; /* Prevents zoom on iOS */
                padding: 12px;
            }}
            
            h1 {{ font-size: 24px; }}
            h2 {{ font-size: 20px; }}
            h3 {{ font-size: 18px; }}
            
            .stat-box-value {{
                font-size: 24px;
            }}
            
            .table {{
                font-size: 12px;
            }}
            
            .table thead th,
            .table tbody td {{
                padding: var(--spacing-sm);
            }}
            
            .table-responsive {{
                margin-bottom: var(--spacing-md);
            }}
        }}
        
        @media (max-width: 480px) {{
            .navbar {{
                padding: var(--spacing-sm) var(--spacing-md);
            }}
            
            .navbar-brand {{
                font-size: 16px;
            }}
            
            .btn {{
                width: 100%;
                margin-bottom: var(--spacing-sm);
            }}
        }}

        /* ===========================
           TVC SPECIFIC STYLES
           =========================== */
        
        .tvc-preset-container {{
            margin-bottom: 10px;
        }}
        
        .tvc-preset-label {{
            margin-right: 10px; 
            font-weight: 600; 
            color: {COLORS["text_secondary"]};
        }}
        
        .tvc-preset-btn {{
            margin-right: 5px !important;
        }}

        /* ===========================
           UX/UI IMPROVEMENTS
           =========================== */
        
        .form-section {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }}
        
        .form-section-title {{
            color: {COLORS["primary"]};
            font-size: 16px;
            margin-bottom: var(--spacing-md);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .form-section-required::before {{
            content: "";
            display: none;
        }}
        
        .form-section-optional::before {{
            content: "";
            display: none;
        }}
        
        .form-section-advanced {{
            background: {COLORS["smoke_white"]};
            border-left: 4px solid {COLORS["warning"]};
        }}
        
        .results-section {{
            margin-top: var(--spacing-2xl);
            padding-top: var(--spacing-xl);
            border-top: 2px solid {COLORS["border"]};
        }}
        
        .results-header {{
            display: flex;
            align-items: center;
            margin-bottom: var(--spacing-lg);
            gap: 12px;
        }}
        
        .results-title {{
            margin: 0;
            color: {COLORS["primary_dark"]};
        }}
        
        .stat-table {{
            font-size: 13px;
        }}
        
        .stat-table th {{
            background-color: {COLORS["smoke_white"]};
            font-weight: 600;
        }}
        
        .stat-table .sig-p {{
            background-color: {COLORS["success"]};
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}
        
        .stat-table td {{
            font-family: var(--font-family-mono);
        }}
        
        .help-icon {{
            display: inline-block;
            width: 18px;
            height: 18px;
            background: {COLORS["primary"]};
            color: white;
            border-radius: 50%;
            text-align: center;
            font-size: 12px;
            cursor: help;
            margin-left: 4px;
            font-weight: 700;
            line-height: 18px;
        }}
        
        .input-help-text {{
            font-size: 12px;
            color: {COLORS["text_secondary"]};
            margin-top: 6px;
            padding: 8px;
            background: {COLORS["smoke_white"]};
            border-left: 3px solid {COLORS["primary"]};
            border-radius: 4px;
            display: block;
        }}
        
        .label-with-help {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .workflow-progress {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 16px;
            background: {COLORS["primary_light"]};
            border-radius: var(--radius-lg);
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        
        .step {{
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            color: {COLORS["text_secondary"]};
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
        }}
        
        .step.active {{
            background: {COLORS["primary"]};
            color: white;
            border-color: {COLORS["primary"]};
        }}
        
        .step-divider {{
            color: {COLORS["neutral"]};
        }}
        
        /* ===========================
           SECTION 7: EMPTY STATE & SKELETON
           =========================== */
        
        /* 7.1 Empty State Handling */
        .empty-state {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 240px;
            background: {COLORS["smoke_white"]};
            border-radius: var(--radius-lg);
            border: 1px dashed var(--color-border);
            text-align: center;
            padding: var(--spacing-xl);
            margin-bottom: var(--spacing-lg);
        }}

        .empty-state-content {{
            max-width: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--spacing-md);
        }}

        .empty-state-icon {{
            font-size: 64px;
            line-height: 1;
            margin-bottom: var(--spacing-md);
            opacity: 0.5;
            filter: grayscale(1);
        }}
        
        .empty-state h3 {{
            font-size: var(--text-heading-size);
            color: var(--color-text);
            margin-bottom: var(--spacing-xs);
        }}
        
        .empty-state p {{
            font-size: var(--text-body-size);
            color: var(--color-text-secondary);
            margin-bottom: var(--spacing-lg);
        }}
        
        /* 7.2 Skeleton Loaders */
        .skeleton-container {{
            width: 100%;
            padding: var(--spacing-md);
        }}

        .skeleton {{
            background: linear-gradient(
                90deg,
                #F0F0F0 0%,
                #E8E8E8 50%,
                #F0F0F0 100%
            );
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
            border-radius: var(--radius-sm);
            margin-bottom: 12px;
        }}

        .skeleton-text {{
            height: 16px;
            width: 80%;
        }}
        
        .skeleton-text-sm {{
            height: 12px;
            width: 60%;
        }}

        .skeleton-chart {{
            height: 300px;
            width: 100%;
            border-radius: var(--radius-md);
        }}
        
        .skeleton-card {{
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            background: white;
            margin-bottom: var(--spacing-lg);
        }}

        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}

        /* ===========================
           RESULT TRANSITIONS
           =========================== */
        .fade-in-entry {{
            animation: fadeInAnimation 0.5s ease-out forwards;
            opacity: 0;
            transform: translateY(10px);
        }}

        @keyframes fadeInAnimation {{
            0% {{
                opacity: 0;
                transform: translateY(10px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            .fade-in-entry {{
                animation: none;
                opacity: 1;
                transform: none;
            }}
        }}

        /* ===========================
           DOWNLOAD STATUS BADGE
           =========================== */
        .download-status-badge {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 11.5px;
            font-weight: 500;
            padding: 3px 0;
            margin-top: 4px;
            line-height: 1.3;
        }}

        .download-status-badge.ready {{
            color: {COLORS["success"]};
        }}

        .download-status-badge.pending {{
            color: {COLORS["text_secondary"]};
            opacity: 0.7;
        }}

        /* ===========================
           CUSTOM REDESIGN CLASSES
           =========================== */
        .muted-placeholder {{
            color: {COLORS["text_secondary"]} !important;
            text-align: center !important;
            padding: var(--spacing-lg) !important;
            font-style: italic !important;
            font-size: 13px !important;
        }}

        .info-callout {{
            padding: var(--spacing-md) !important;
            border-radius: var(--radius-md) !important;
            background-color: {COLORS["primary_light"]} !important;
            border-left: 3px solid {COLORS["border"]} !important;
            font-size: 14px !important;
            color: {COLORS["text"]} !important;
        }}

        .text-muted-sm {{
            font-size: 13px !important;
            color: {COLORS["text_secondary"]} !important;
        }}

        .home-grid {{
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)) !important;
            gap: 20px !important;
        }}

        .forest-plot-section, .assumptions-section {{
            margin-top: 32px !important;
            padding: 16px 0 0 0 !important;
            border-top: 1px solid {COLORS["border"]} !important;
        }}
    </style>
    """

    return css


def style_card_header(title: str, icon: str = "") -> str:
    """
    Generate styled card header HTML.

    Args:
        title: Header text
        icon: Optional emoji or icon

    Returns:
        HTML string for card header
    """
    COLORS = get_color_palette()
    return f"""
    <div style="
        background: transparent;
        border-bottom: 1px solid {COLORS["border"]};
        padding: 14px 16px;
        font-weight: 500;
        color: {COLORS["text"]};
        border-radius: 6px 6px 0 0;
        font-size: 14px;
    ">
        {icon} {title}
    </div>
    """


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB values for rgba()."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "0, 0, 0"  # Fallback for invalid input
    try:
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    except ValueError:
        return "0, 0, 0"  # Fallback for non-hex characters
    return f"{r}, {g}, {b}"


def style_status_badge(status: str, text: str) -> str:
    """
    Generate styled status badge.

    Args:
        status: 'success', 'danger', 'warning', 'info'
        text: Badge text

    Returns:
        HTML string for status badge
    """
    COLORS = get_color_palette()
    color_map = {
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "info": COLORS["info"],
    }
    color = color_map.get(status, COLORS["info"])
    text_color = "#000" if status == "warning" else color

    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 6px;
        background-color: rgba({_hex_to_rgb(color)}, 0.1);
        color: {text_color};
        border: 1px solid {color};
        font-weight: 600;
        font-size: 12px;
    ">
        {text}
    </span>
    """


def style_alert(alert_type: str, message: str, title: str = "") -> str:
    """
    Generate styled alert/notification box.

    Args:
        alert_type: 'success', 'danger', 'warning', 'info'
        message: Alert message
        title: Optional alert title

    Returns:
        HTML string for alert
    """
    COLORS = get_color_palette()
    color_map = {
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "info": COLORS["primary"],
    }
    color = color_map.get(alert_type, COLORS["info"])
    icon_map = {
        "success": "✅",
        "danger": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    icon = icon_map.get(alert_type, "")

    title_html = f"<strong>{title}</strong><br>" if title else ""

    return f"""
    <div style="
        background-color: rgba({_hex_to_rgb(color)}, 0.1);
        border: 1px solid {color};
        border-radius: 6px;
        padding: 12px 16px;
        color: {color};
        margin-bottom: 16px;
    ">
        {icon} {title_html}{message}
    </div>
    """


def get_color_code(color_name: str) -> str:
    """
    Get hex color code by name.

    Args:
        color_name: Color name (e.g., 'primary', 'success')

    Returns:
        Hex color code
    """
    colors = get_color_palette()
    return colors.get(color_name, "#1E3A5F")
