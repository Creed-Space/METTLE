<script lang="ts">
	import { MettleHeader, MettleFooter } from '$lib/components';
	import '../app.css';

	let { children } = $props();
</script>

<a href="#main-content" class="skip-link">Skip to main content</a>
<MettleHeader />
<main id="main-content">
	{@render children()}
</main>
<MettleFooter />
