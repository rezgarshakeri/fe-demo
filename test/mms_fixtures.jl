# Manufactured solution (trigonometric), shared by the Dirichlet and traction
# elasticity tests (notebooks/Elasticity2D.ipynb).
function uex(x, y)
    A0 = 1e-3
    s = pi * 2
    x0 = pi / 2
    y0 = x0
    ux = @. A0 * sin(s * x - x0) * sin(s * y - y0)
    uy = @. 2 * ux
    return [ux; uy]
end

function f(lambda, mu, x, y)
    A0 = 1e-3
    s = pi * 2
    x0 = pi / 2
    y0 = x0
    ux = @. A0 * sin(s * x - x0) * sin(s * y - y0)

    ux_xx = @. -s^2 * ux
    ux_yy = @. -s^2 * ux
    uy_xx = @. 2 * ux_xx
    uy_yy = @. 2 * ux_yy
    ux_xy = @. A0 * s^2 * cos(s * x - x0) * cos(s * y - y0)
    uy_yx = @. 2 * ux_xy

    fx = @. -mu * (ux_xx + ux_yy) - (mu + lambda) * (ux_xx + uy_yx)
    fy = @. -mu * (uy_xx + uy_yy) - (mu + lambda) * (ux_xy + uy_yy)
    return [fx; fy]
end

# sigma(uex), so we can prescribe the exact traction t = sigma(uex) * n on a face.
function sigma_exact(lambda, mu, x, y)
    A0 = 1e-3
    s = pi * 2
    x0 = pi / 2
    y0 = x0
    ux_x = @. A0 * s * cos(s * x - x0) * sin(s * y - y0)
    ux_y = @. A0 * s * sin(s * x - x0) * cos(s * y - y0)
    uy_x = @. 2 * ux_x
    uy_y = @. 2 * ux_y
    tr_e = ux_x + uy_y
    s11 = @. lambda * tr_e + 2 * mu * ux_x
    s12 = @. mu * (ux_y + uy_x)
    return s11, s12
end

f_zero(lambda, mu, x, y) = zeros(2 * length(x)) # zero body force, shared by hyperelasticity/plasticity tests
