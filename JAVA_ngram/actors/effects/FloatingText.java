
	public FloatingText() {
		super(9*PixelScene.defaultZoom);
		setHightlighting(false);
	}
	
	@Override
	public void update() {
		super.update();
		
		if (timeLeft >= 0) {
			if ((timeLeft -= Game.elapsed) <= 0) {
				kill();
			} else {
				float p = timeLeft / LIFESPAN;
				alpha( p > 0.5f ? 1 : p * 2 );
				
				float yMove = (DISTANCE / LIFESPAN) * Game.elapsed;
				y -= yMove;
				for (RenderedText t : words){
					t.y -= yMove;
				}

				if (icon != null){
					icon.alpha(p > 0.5f ? 1 : p * 2);
					icon.y -= yMove;
				}
			}
		}
	}

	@Override
	protected synchronized void layout() {
		super.layout();
		if (icon != null){
			if (iconLeft){
				icon.x = left();
			} else {
				icon.x = left() + width() - icon.width();
			}
			icon.y = top();
			PixelScene.align(icon);
		}
	}

	@Override
	public float width() {
		float width = super.width();
		if (icon != null){
			width += icon.width()-0.5f;
		}
		return width;
	}

	@Override
	public void kill() {
		if (key != -1) {
			synchronized (stacks) {
				stacks.get(key).remove(this);
			}
			key = -1;
		}
		super.kill();
	}
	
	@Override
	public void destroy() {
		kill();
		super.destroy();
	}
	
	public void reset( float x, float y, String text, int color, int iconIdx, boolean left ) {
		
		revive();
		
		zoom( 1 / (float)PixelScene.defaultZoom );

		text( text );
		hardlight( color );

		if (iconIdx != NO_ICON){
			icon = new Image( Assets.Effects.TEXT_ICONS);
			icon.frame(iconFilm.get(iconIdx));
			add(icon);
			iconLeft = left;
			if (iconLeft){
				align(RIGHT_ALIGN);
			}
		} else {
			icon = null;
		}

		setPos(
			PixelScene.align( Camera.main, x - width() / 2),
			PixelScene.align( Camera.main, y - height())
		);
		
		timeLeft = LIFESPAN;
	}
	
	/* STATIC METHODS */

	public static void show( float x, float y, String text, int color) {
		show(x, y, -1, text, color, -1, false);
	}
	
	public static void show( float x, float y, int key, String text, int color) {
		show(x, y, key, text, color, -1, false);
	}
	
	public static void show( float x, float y, int key, String text, int color, int iconIdx, boolean left ) {
		Game.runOnRenderThread(new Callback() {
			@Override
			public void call() {
				FloatingText txt = GameScene.status();
				if (txt != null){
					txt.reset(x, y, text, color, iconIdx, left);
					if (key != -1) push(txt, key);
				}
			}
		});
	}
	
	private static void push( FloatingText txt, int key ) {
		
		synchronized (stacks) {
			txt.key = key;
			
			ArrayList<FloatingText> stack = stacks.get(key);
			if (stack == null) {
				stack = new ArrayList<>();
				stacks.put(key, stack);
			}
			
			if (stack.size() > 0) {
				FloatingText below = txt;
				int aboveIndex = stack.size() - 1;
				int numBelow = 0;
				while (aboveIndex >= 0) {
					numBelow++;
					FloatingText above = stack.get(aboveIndex);
					if (above.bottom() + 4 > below.top()) {
						above.setPos(above.left(), below.top() - above.height() - 4);

						//reduce remaining time on texts being nudged up, to prevent spam
						above.timeLeft = Math.min(above.timeLeft, LIFESPAN-(numBelow/5f));
						above.timeLeft = Math.max(above.timeLeft, 0);
						
						below = above;
						aboveIndex--;
					} else {
						break;
					}
				}
			}
			
			stack.add(txt);
		}
	}
}
